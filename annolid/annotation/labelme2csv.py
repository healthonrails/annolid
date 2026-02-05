import csv
import os
import re
from pathlib import Path
from tqdm import tqdm
import argparse
from annolid.utils.annotation_compat import shape_to_mask
from annolid.utils.shapes import masks_to_bboxes, polygon_center
from annolid.annotation.masks import binary_mask_to_coco_rle
from annolid.annotation.timestamps import convert_frame_number_to_time
from annolid.utils.annotation_store import (
    AnnotationStore,
    AnnotationStoreError,
    load_labelme_json,
)


def read_json_file(file_path):
    """
    Read JSON data from a file.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - dict: JSON data.
    """
    try:
        return load_labelme_json(file_path)
    except (FileNotFoundError, AnnotationStoreError, ValueError):
        return {}


def get_frame_number_from_filename(file_name):
    """
    Extract frame number from a JSON file name.

    Parameters:
    - file_name (str): JSON file name.

    Returns:
    - int: Extracted frame number.
    """
    return int(file_name.split("_")[-1].replace(".json", ""))


def _normalize_fps(value):
    try:
        fps_value = float(value)
    except (TypeError, ValueError):
        return None
    return fps_value if fps_value > 0 else None


def _find_video_for_json_folder(json_folder_path: Path):
    stem = json_folder_path.name
    parent = json_folder_path.parent
    for ext in (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg"):
        candidate = parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def convert_shape_to_mask(img_shape, points):
    """
    Convert shape points to a binary mask.

    Parameters:
    - img_shape (tuple): Image shape (width, height).
    - points (list): List of points.

    Returns:
    - ndarray: Binary mask.
    """
    mask = shape_to_mask(
        img_shape, points, shape_type=None, line_width=10, point_size=5
    )
    return mask


def _extract_frame_number_from_json_name(file_name: str) -> int | None:
    match = re.search(r"(\d+)(?=\.json$)", file_name)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _collect_annotation_files(json_folder_path: Path) -> list[str]:
    json_files = sorted(f for f in os.listdir(json_folder_path) if f.endswith(".json"))
    store = AnnotationStore.for_frame_path(
        json_folder_path / f"{json_folder_path.name}_000000000.json"
    )
    if store.store_path.exists():
        store_files = [
            f"{json_folder_path.name}_{frame:09d}.json"
            for frame in sorted(store.iter_frames())
        ]
        json_files = sorted(set(json_files) | set(store_files))
    return json_files


def _existing_csv_covers_all_frames(csv_path: Path, required_frames: set[int]) -> bool:
    if not csv_path.exists() or not required_frames:
        return False

    try:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                return False
            try:
                frame_idx = header.index("frame_number")
            except ValueError:
                return False
            seen_frames = set()
            for row in reader:
                if frame_idx >= len(row):
                    continue
                raw = str(row[frame_idx]).strip()
                if not raw:
                    continue
                try:
                    seen_frames.add(int(float(raw)))
                except ValueError:
                    continue
    except OSError:
        return False

    return required_frames.issubset(seen_frames)


def convert_json_to_csv(
    json_folder,
    csv_file=None,
    progress_callback=None,
    stop_event=None,
    tracked_csv_file=None,
    fps=None,
):
    """
    Convert JSON files in a folder to annolid CSV format file.

    Parameters:
    - json_folder (str): Path to the folder containing JSON files.
    - csv_file (str): Output CSV file.
    - progress_callback (function): Callback function to report progress.
    - fps (float): Frames per second used for timestamp estimation.


    Returns:
    - None
    """
    csv_header = [
        "frame_number",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
        "instance_name",
        "class_score",
        "segmentation",
        "tracking_id",
    ]

    if csv_file is None:
        csv_file = json_folder + "_tracking.csv"

    def _normalize_tracking_id(value):
        if value is None:
            return 0
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            try:
                if value.is_integer():
                    return int(value)
            except Exception:
                return 0
        if isinstance(value, str):
            raw = value.strip()
            if raw.isdigit():
                return int(raw)
        return 0

    def _shape_instance_name(shape, *, shape_type: str):
        label = shape.get("label")
        instance_label = shape.get("instance_label") or shape.get("instance_name")
        label_text = label.strip() if isinstance(label, str) and label.strip() else ""
        instance_text = (
            instance_label.strip()
            if isinstance(instance_label, str) and instance_label.strip()
            else ""
        )
        if shape_type == "point":
            if instance_text and label_text:
                return f"{instance_text}:{label_text}"
            if not instance_text and label_text:
                group_id = shape.get("group_id")
                if group_id not in (None, ""):
                    return f"{group_id}:{label_text}"
            return label_text or instance_text
        return instance_text or label_text

    def _shape_score(shape):
        for key in ("score", "class_score", "confidence"):
            value = shape.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 1.0

    def _report_progress(progress):
        if progress_callback is None:
            return
        try:
            progress_callback(progress)
        except Exception:
            return

    tracked_output = None
    tracked_writer = None
    tracked_seen = set()
    tracked_header = [
        "frame_number",
        "instance_name",
        "cx",
        "cy",
        "motion_index",
        "timestamps",
    ]
    fps_value = _normalize_fps(fps)
    video_fps_checked = False
    if tracked_csv_file:
        try:
            tracked_output = open(tracked_csv_file, "w", newline="")
            tracked_writer = csv.writer(tracked_output)
            tracked_writer.writerow(tracked_header)
        except OSError:
            tracked_output = None
            tracked_writer = None

    json_folder_path = Path(json_folder)
    json_files = _collect_annotation_files(json_folder_path)
    if not json_files:
        return f"No annotation files found in {json_folder}."

    required_frames = {
        frame
        for frame in (_extract_frame_number_from_json_name(name) for name in json_files)
        if frame is not None
    }
    csv_path = Path(csv_file)
    if _existing_csv_covers_all_frames(csv_path, required_frames):
        if tracked_output is not None:
            try:
                tracked_output.close()
            except Exception:
                pass
        _report_progress(100)
        return str(csv_path)

    video_path = None
    if tracked_writer is not None:
        video_path = _find_video_for_json_folder(json_folder_path)

    try:
        with open(csv_file, "w", newline="") as csv_output:
            csv_writer = csv.writer(csv_output)
            csv_writer.writerow(csv_header)

            total_files = len(json_files)
            num_processed_files = 0

            for json_file in tqdm(
                json_files, desc="Converting JSON files", unit="files"
            ):
                if stop_event is not None and stop_event.is_set():
                    return "Stopped"
                json_path = os.path.join(json_folder, json_file)
                data = read_json_file(json_path)
                if not data:
                    num_processed_files += 1
                    progress = int((num_processed_files / total_files) * 100)
                    _report_progress(progress)
                    continue

                try:
                    frame_number = get_frame_number_from_filename(json_file)
                except ValueError:
                    # Skip metadata files that do not follow the frame naming scheme
                    num_processed_files += 1
                    progress = int((num_processed_files / total_files) * 100)
                    _report_progress(progress)
                    continue
                img_height = data.get("imageHeight")
                img_width = data.get("imageWidth")
                shapes = data.get("shapes") or []

                if img_height is None or img_width is None:
                    num_processed_files += 1
                    progress = int((num_processed_files / total_files) * 100)
                    _report_progress(progress)
                    continue

                img_shape = (img_height, img_width)
                timestamp_value = ""
                if tracked_writer is not None:
                    if fps_value is None:
                        fps_value = _normalize_fps(
                            data.get("fps")
                            or data.get("video_fps")
                            or (data.get("flags") or {}).get("fps")
                        )
                    if fps_value is None and not video_fps_checked:
                        video_fps_checked = True
                        if video_path is not None:
                            try:
                                from annolid.data.videos import get_video_fps

                                fps_value = _normalize_fps(
                                    get_video_fps(str(video_path))
                                )
                            except Exception:
                                fps_value = None
                    if fps_value is None:
                        fps_value = 29.97
                    timestamp_value = convert_frame_number_to_time(
                        frame_number, fps_value
                    )

                for shape in shapes:
                    if not isinstance(shape, dict):
                        continue
                    shape_type = (shape.get("shape_type") or "").lower()
                    instance_name = _shape_instance_name(shape, shape_type=shape_type)
                    points = shape.get("points") or []
                    if shape_type == "rectangle":
                        if len(points) < 2:
                            continue
                        a, b = points[0], points[1]
                        if (
                            not isinstance(a, (list, tuple))
                            or not isinstance(b, (list, tuple))
                            or len(a) < 2
                            or len(b) < 2
                        ):
                            continue
                        x1 = min(float(a[0]), float(b[0]))
                        y1 = min(float(a[1]), float(b[1]))
                        x2 = max(float(a[0]), float(b[0]))
                        y2 = max(float(a[1]), float(b[1]))
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        class_score = _shape_score(shape)
                        tracking_id = _normalize_tracking_id(shape.get("group_id"))
                        segmentation = ""

                        csv_writer.writerow(
                            [
                                frame_number,
                                x1,
                                y1,
                                x2,
                                y2,
                                cx,
                                cy,
                                instance_name,
                                class_score,
                                segmentation,
                                tracking_id,
                            ]
                        )
                        if tracked_writer is not None and instance_name:
                            motion_index = shape.get("motion_index")
                            if motion_index is None:
                                description = shape.get("description") or ""
                                match = re.search(
                                    r"motion_index\s*[:=]\s*([-0-9.eE]+)", description
                                )
                                if match:
                                    try:
                                        motion_index = float(match.group(1))
                                    except ValueError:
                                        motion_index = None
                            try:
                                motion_index = (
                                    float(motion_index)
                                    if motion_index is not None
                                    else -1
                                )
                            except (TypeError, ValueError):
                                motion_index = -1
                            key = (frame_number, instance_name)
                            if key not in tracked_seen:
                                tracked_seen.add(key)
                                tracked_writer.writerow(
                                    [
                                        frame_number,
                                        instance_name,
                                        cx,
                                        cy,
                                        motion_index,
                                        timestamp_value,
                                    ]
                                )
                        continue

                    if shape_type == "point":
                        if not points:
                            continue
                        point = points[0]
                        if not isinstance(point, (list, tuple)) or len(point) < 2:
                            continue
                        x = float(point[0])
                        y = float(point[1])
                        class_score = _shape_score(shape)
                        tracking_id = _normalize_tracking_id(shape.get("group_id"))
                        csv_writer.writerow(
                            [
                                frame_number,
                                x,
                                y,
                                x,
                                y,
                                x,
                                y,
                                instance_name,
                                class_score,
                                "",
                                tracking_id,
                            ]
                        )
                        if tracked_writer is not None and instance_name:
                            motion_index = shape.get("motion_index")
                            if motion_index is None:
                                description = shape.get("description") or ""
                                match = re.search(
                                    r"motion_index\s*[:=]\s*([-0-9.eE]+)", description
                                )
                                if match:
                                    try:
                                        motion_index = float(match.group(1))
                                    except ValueError:
                                        motion_index = None
                            try:
                                motion_index = (
                                    float(motion_index)
                                    if motion_index is not None
                                    else -1
                                )
                            except (TypeError, ValueError):
                                motion_index = -1
                            key = (frame_number, instance_name)
                            if key not in tracked_seen:
                                tracked_seen.add(key)
                                tracked_writer.writerow(
                                    [
                                        frame_number,
                                        instance_name,
                                        x,
                                        y,
                                        motion_index,
                                        timestamp_value,
                                    ]
                                )
                        continue

                    if len(points) > 2:
                        mask = convert_shape_to_mask(img_shape, points)
                        bboxs = masks_to_bboxes(mask[None, :, :])
                        segmentation = binary_mask_to_coco_rle(mask)
                        class_score = _shape_score(shape)
                        tracking_id = _normalize_tracking_id(shape.get("group_id"))

                        if len(bboxs) > 0:
                            cx, cy = polygon_center(points)
                            x1, y1, x2, y2 = bboxs[0]
                            csv_writer.writerow(
                                [
                                    frame_number,
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    cx,
                                    cy,
                                    instance_name,
                                    class_score,
                                    segmentation,
                                    tracking_id,
                                ]
                            )
                            if tracked_writer is not None and instance_name:
                                motion_index = shape.get("motion_index")
                                if motion_index is None:
                                    description = shape.get("description") or ""
                                    match = re.search(
                                        r"motion_index\s*[:=]\s*([-0-9.eE]+)",
                                        description,
                                    )
                                    if match:
                                        try:
                                            motion_index = float(match.group(1))
                                        except ValueError:
                                            motion_index = None
                                try:
                                    stored_cx = float(shape.get("cx"))
                                    stored_cy = float(shape.get("cy"))
                                except (TypeError, ValueError):
                                    stored_cx = cx
                                    stored_cy = cy
                                try:
                                    motion_index = (
                                        float(motion_index)
                                        if motion_index is not None
                                        else -1
                                    )
                                except (TypeError, ValueError):
                                    motion_index = -1
                                key = (frame_number, instance_name)
                                if key in tracked_seen:
                                    continue
                                tracked_seen.add(key)
                                tracked_writer.writerow(
                                    [
                                        frame_number,
                                        instance_name,
                                        stored_cx,
                                        stored_cy,
                                        motion_index,
                                        timestamp_value,
                                    ]
                                )
                num_processed_files += 1
                progress = int((num_processed_files / total_files) * 100)
                _report_progress(progress)
    finally:
        if tracked_writer is not None:
            try:
                tracked_output.close()
            except Exception:
                pass

    if tracked_csv_file:
        try:
            from annolid.postprocessing.video_timestamp_annotator import (
                annotate_csv,
            )

            video_path = _find_video_for_json_folder(Path(json_folder))
            if video_path is not None and video_path.exists():
                annotate_csv(Path(tracked_csv_file), video_path)
        except Exception:
            pass

    return str(csv_file)


def main():
    """
    Main function to parse command-line arguments and execute the conversion.
    """
    parser = argparse.ArgumentParser(description="Convert JSON files to a CSV file.")
    parser.add_argument(
        "--json_folder", required=True, help="Path to the folder containing JSON files."
    )
    parser.add_argument("--csv_file", required=False, help="Output CSV file.")

    args = parser.parse_args()
    convert_json_to_csv(args.json_folder, args.csv_file)


if __name__ == "__main__":
    main()
