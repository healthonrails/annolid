import csv
import os
import cv2
import torch
import gdown
import json
import numpy as np
from bisect import bisect_right
from datetime import datetime, timedelta
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from PIL import Image
from annolid.gui.shape import MaskShape, Shape
from annolid.annotation.keypoints import save_labels
from annolid.segmentation.cutie_vos.interactive_utils import (
    image_to_torch,
    torch_prob_to_numpy_mask,
    index_numpy_to_one_hot_torch,
    overlay_davis,
    color_id_mask,
)
from omegaconf import open_dict
from hydra import compose, initialize
from annolid.segmentation.cutie_vos.model.cutie import CUTIE
from annolid.segmentation.cutie_vos.inference.inference_core import InferenceCore
from annolid.utils.annotation_compat import shapes_to_label
from annolid.utils.shapes import extract_flow_points_in_mask
from annolid.utils.devices import get_device
from annolid.utils.logger import logger
from annolid.motion.optical_flow import (
    compute_optical_flow,
    optical_flow_compute_kwargs,
    optical_flow_settings_from,
)
from annolid.utils import draw
from annolid.utils.lru_cache import BboxCache
from annolid.utils.annotation_store import AnnotationStore
from hydra.core.global_hydra import GlobalHydra

"""
References:
@inproceedings{cheng2023putting,
  title={Putting the Object Back into Video Object Segmentation},
  author={Cheng, Ho Kei and Oh, Seoung Wug and Price, Brian and Lee, Joon-Young and Schwing, Alexander},
  booktitle={arXiv},
  year={2023}
}
https://github.com/hkchengrex/Cutie/tree/main
"""


def find_mask_center_opencv(mask):
    # Convert boolean mask to integer mask (0 for background, 255 for foreground)
    mask_int = mask.astype(np.uint8) * 255

    # Calculate the moments of the binary image
    moments = cv2.moments(mask_int)

    # Calculate the centroid coordinates
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    return cx, cy


@dataclass(frozen=True)
class SeedFrame:
    frame_index: int
    png_path: Path
    json_path: Path


@dataclass
class SeedSegment:
    seed: Optional[SeedFrame]
    start_frame: int
    end_frame: Optional[int]
    mask: np.ndarray
    labels_map: Dict[str, int]
    active_labels: List[str]


class CutieCoreVideoProcessor:
    _REMOTE_MODEL_URL = (
        "https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth"
    )
    _MD5 = "a6071de6136982e396851903ab4c083a"
    _DISCOVERED_SEEDS_CACHE: Dict[str, List[SeedFrame]] = {}

    def __init__(self, video_name, *args, **kwargs):
        self.video_name = video_name
        results_folder = kwargs.get("results_folder")
        self.video_folder = (
            Path(results_folder) if results_folder else Path(video_name).with_suffix("")
        )
        self.results_folder = self.video_folder
        self.mem_every = kwargs.get("mem_every", 5)
        self.debug = kwargs.get("debug", False)
        # T_max parameter default 5
        self.max_mem_frames = kwargs.get("t_max_value", 5)
        self.use_cpu_only = kwargs.get("use_cpu_only", False)
        self.epsilon_for_polygon = kwargs.get("epsilon_for_polygon", 2.0)
        self.processor = None
        self.num_tracking_instances = 0
        current_file_path = os.path.abspath(__file__)
        self.current_folder = os.path.dirname(current_file_path)
        self.device = "cpu" if self.use_cpu_only else get_device()
        logger.info(f"Running device: {self.device}.")
        self.cutie, self.cfg = self._initialize_model()
        logger.info(
            f"Using epsilon: {self.epsilon_for_polygon} for polygon approximation."
        )

        self.output_tracking_csvpath = None
        self._frame_number = None
        self._motion_index = ""
        self._instance_name = ""
        self._flow = None
        self._flow_hsv = None
        self._mask = None
        self.cache = BboxCache(max_size=self.mem_every * 10)
        self.sam_hq = None
        self.output_tracking_csvpath = str(self.video_folder) + "_tracked.csv"
        self.showing_KMedoids_in_mask = False
        self._optical_flow_settings = optical_flow_settings_from(kwargs)
        self.compute_optical_flow = bool(
            self._optical_flow_settings.get("compute_optical_flow", False)
        )
        self.optical_flow_backend = str(
            self._optical_flow_settings.get("optical_flow_backend", "farneback")
        )
        self._optical_flow_kwargs = optical_flow_compute_kwargs(
            self._optical_flow_settings
        )
        self.auto_missing_instance_recovery = kwargs.get(
            "auto_missing_instance_recovery", False
        )
        self.continue_on_missing_instances = bool(
            kwargs.get("continue_on_missing_instances", True)
        )
        logger.info(
            f"Auto missing instance recovery is set to {self.auto_missing_instance_recovery}."
        )
        self._seed_frames: List[SeedFrame] = []
        self.label_registry: Dict[str, int] = {"_background_": 0}
        self._seed_segment_lookup: Dict[int, SeedSegment] = {}
        self._committed_seed_frames: Set[int] = set()
        self._cached_labeled_frames: Optional[Set[int]] = None
        self._last_mask_area_ratio: Dict[str, float] = {}
        self.reject_suspicious_mask_jumps = bool(
            kwargs.get("reject_suspicious_mask_jumps", False)
        )
        self._seed_object_counts: Dict[int, int] = {}
        self._global_object_ids: List[int] = []
        self._global_label_names: Dict[int, str] = {}
        self._video_seed_cache: Dict[str, Dict[str, Any]] = {}
        self._current_video_cache_key: Optional[str] = None
        self._video_active_object_ids: Set[int] = set()
        # Latest per-instance masks available in the current run.
        self._recent_instance_masks: Dict[str, np.ndarray] = {}
        self._recent_instance_mask_frames: Dict[str, int] = {}
        self._last_saved_instance_masks: Dict[str, np.ndarray] = {}
        self._repetitive_warning_state: Dict[Tuple[str, str], Dict[str, int]] = {}

    def set_same_hq(self, sam_hq):
        self.sam_hq = sam_hq

    def _should_stop(self, pred_worker=None) -> bool:
        if pred_worker is None:
            return False
        try:
            if hasattr(pred_worker, "is_stopped") and pred_worker.is_stopped():
                return True
        except Exception:
            pass
        try:
            stop_event = getattr(pred_worker, "stop_event", None)
            if stop_event is not None and stop_event.is_set():
                return True
        except Exception:
            pass
        try:
            from qtpy import QtCore

            thread = QtCore.QThread.currentThread()
            if thread is not None and thread.isInterruptionRequested():
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _normalize_tracking_scalar(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(parsed):
            return default
        return parsed

    def _log_repetitive_warning(
        self,
        key: Tuple[str, str],
        message: str,
        *,
        every: int = 25,
    ) -> None:
        """Log repetitive warnings with first-hit + periodic cadence."""
        state = getattr(self, "_repetitive_warning_state", None)
        if not isinstance(state, dict):
            state = {}
            self._repetitive_warning_state = state

        entry = state.get(key)
        current_frame = int(getattr(self, "_frame_number", -1))
        if entry is None:
            entry = {"streak": 0, "suppressed": 0, "last_frame": current_frame}
        else:
            last_frame = int(entry.get("last_frame", current_frame))
            if current_frame != last_frame + 1:
                entry["streak"] = 0
                entry["suppressed"] = 0
            entry["last_frame"] = current_frame

        entry["streak"] = int(entry.get("streak", 0)) + 1
        should_log = entry["streak"] == 1 or (every > 0 and entry["streak"] % every == 0)
        if should_log:
            suppressed = int(entry.get("suppressed", 0))
            if suppressed > 0:
                logger.warning(
                    "Suppressed %s repetitive warning(s) for %s/%s.",
                    suppressed,
                    key[0],
                    key[1],
                )
                entry["suppressed"] = 0
            logger.warning(message)
        else:
            entry["suppressed"] = int(entry.get("suppressed", 0)) + 1

        state[key] = entry

    def _read_tracking_rows_from_csv(self, path: Path) -> Dict[tuple, tuple]:
        rows: Dict[tuple, tuple] = {}
        try:
            with open(path, "r", newline="") as csv_input:
                reader = csv.DictReader(csv_input)
                for row in reader:
                    frame_raw = row.get("frame_number")
                    instance_name = row.get("instance_name")
                    if frame_raw is None or instance_name is None:
                        continue
                    try:
                        frame_number = int(float(frame_raw))
                    except (TypeError, ValueError):
                        continue
                    cx = self._normalize_tracking_scalar(row.get("cx"), default=0.0)
                    cy = self._normalize_tracking_scalar(row.get("cy"), default=0.0)
                    motion_index = self._normalize_tracking_scalar(
                        row.get("motion_index"), default=-1.0
                    )
                    rows[(frame_number, instance_name)] = (
                        frame_number,
                        instance_name,
                        cx,
                        cy,
                        motion_index,
                    )
        except FileNotFoundError:
            return rows
        except Exception as exc:
            logger.error("Failed to read tracking CSV %s: %s", path, exc)
        return rows

    def _read_tracking_rows_from_annotations_store(self) -> Dict[tuple, tuple]:
        rows: Dict[tuple, tuple] = {}
        frame_path = self.video_folder / f"{self.video_folder.name}_000000000.json"
        store = AnnotationStore.for_frame_path(frame_path)
        try:
            for frame_number in sorted(store.iter_frames()):
                try:
                    frame_idx = int(frame_number)
                except (TypeError, ValueError):
                    continue
                record = store.get_frame(frame_idx)
                if record is None:
                    # Some legacy stores may keep string frame keys.
                    record = store.get_frame(str(frame_idx))
                if not isinstance(record, dict):
                    continue
                for shape in record.get("shapes") or []:
                    if not isinstance(shape, dict):
                        continue
                    instance_name = str(shape.get("label") or "").strip()
                    if not instance_name:
                        continue
                    # CUTIE shapes usually store cx/cy/motion_index as top-level fields.
                    # Keep compatibility with nested other_data/otherData payloads.
                    source: Dict[str, Any] = {}
                    for candidate in (
                        shape,
                        shape.get("other_data"),
                        shape.get("otherData"),
                    ):
                        if not isinstance(candidate, dict):
                            continue
                        if any(k in candidate for k in ("cx", "cy", "motion_index")):
                            source = candidate
                            break
                    if not source:
                        continue
                    cx = self._normalize_tracking_scalar(
                        source.get("cx"), default=0.0
                    )
                    cy = self._normalize_tracking_scalar(
                        source.get("cy"), default=0.0
                    )
                    motion_index = self._normalize_tracking_scalar(
                        source.get("motion_index"), default=-1.0
                    )
                    rows[(frame_idx, instance_name)] = (
                        frame_idx,
                        instance_name,
                        cx,
                        cy,
                        motion_index,
                    )
        except Exception as exc:
            logger.error("Failed to read tracking rows from annotation store: %s", exc)
        return rows

    def _write_tracking_csv(self, fps: float) -> None:
        if not self.output_tracking_csvpath:
            return
        output_path = Path(self.output_tracking_csvpath)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        header = [
            "frame_number",
            "instance_name",
            "cx",
            "cy",
            "motion_index",
            "timestamps",
        ]
        # Best-effort cleanup of stale temp files from interrupted prior runs.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception as exc:
            logger.debug("Failed to remove stale tracking tmp file: %s", exc)

        rows: Dict[tuple, tuple] = {}
        if output_path.exists():
            rows.update(self._read_tracking_rows_from_csv(output_path))
        rows.update(self._read_tracking_rows_from_annotations_store())
        if not rows:
            logger.info("No tracking rows to write for CUTIE.")
            return

        try:
            with open(tmp_path, "w", newline="") as csv_output:
                writer = csv.writer(csv_output)
                writer.writerow(header)
                for _, row in sorted(
                    rows.items(), key=lambda item: (item[0][0], item[0][1])
                ):
                    frame_number, instance_name, cx, cy, motion_index = row
                    if fps and fps > 0:
                        timestamp = (
                            datetime(1970, 1, 1)
                            + timedelta(seconds=float(frame_number) / fps)
                        ).time()
                    else:
                        timestamp = ""
                    writer.writerow(
                        [
                            frame_number,
                            instance_name,
                            cx,
                            cy,
                            motion_index,
                            timestamp,
                        ]
                    )
            tmp_path.replace(output_path)
        except Exception as exc:
            logger.error("Failed to write tracking CSV: %s", exc, exc_info=True)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            return
        finally:
            # In case replace succeeded or failed mid-way, ensure no temp residue remains.
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

        try:
            from annolid.postprocessing.video_timestamp_annotator import (
                annotate_csv,
            )

            annotate_csv(output_path, Path(self.video_name))
        except Exception as exc:
            logger.error("Failed to annotate tracking CSV: %s", exc)

    @staticmethod
    def _json_has_manual_seed_content(json_path: Path) -> bool:
        """Return True when a JSON likely represents a user-authored seed.

        Model-produced CUTIE polygons typically carry `motion_index:` descriptions.
        Treat those as non-seed candidates unless there is at least one polygon-like
        shape without this synthetic description.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as fp:
                payload = json.load(fp) or {}
        except Exception:
            return False

        shapes = payload.get("shapes") or []
        has_polygon_like = False
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            shape_type = str(shape.get("shape_type") or "").strip().lower()
            if shape_type not in {"polygon", "rectangle", "circle"}:
                continue
            points = shape.get("points") or []
            if len(points) < 3 and shape_type == "polygon":
                continue
            has_polygon_like = True
            description = str(shape.get("description") or "").strip().lower()
            if not description.startswith("motion_index:"):
                return True

        # If there are no polygon-like shapes this is not a valid seed;
        # if all polygon-like shapes are auto motion-index outputs, skip it.
        return False if has_polygon_like else False

    @staticmethod
    def discover_seed_frames(
        video_name, results_folder: Optional[Path] = None, *, force_refresh: bool = False
    ) -> List[SeedFrame]:
        """Discover seed frame pairs (PNG+JSON) within the video results folder."""
        results_dir = (
            Path(results_folder) if results_folder else Path(video_name).with_suffix("")
        )
        cache_key = str(results_dir.resolve())
        cached = None if force_refresh else CutieCoreVideoProcessor._DISCOVERED_SEEDS_CACHE.get(cache_key)
        if cached is not None:
            logger.info(
                "Using cached CUTIE seeds (%d) for %s", len(cached), results_dir
            )
            return list(cached)
        if not results_dir.exists() or not results_dir.is_dir():
            logger.info(
                f"CUTIE seed discovery skipped: folder does not exist -> {results_dir}"
            )
            return []

        stem = results_dir.name
        stem_prefix = f"{stem}_"
        stem_prefix_lower = stem_prefix.lower()

        def collect_seeds(directory: Path) -> Dict[int, SeedFrame]:
            collected: Dict[int, SeedFrame] = {}
            if not directory.exists() or not directory.is_dir():
                logger.debug(f"Seed scan skipped: not a directory -> {directory}")
                return collected

            png_candidates = sorted(directory.glob("*.png"))
            logger.info(
                f"Seed scan in {directory} found {len(png_candidates)} png candidate(s)"
            )

            for png_path in png_candidates:
                name_lower = png_path.stem.lower()
                if not name_lower.startswith(stem_prefix_lower):
                    logger.debug(f"Skipping {png_path.name}: stem prefix mismatch")
                    continue

                suffix = name_lower[len(stem_prefix_lower) :]
                if len(suffix) != 9 or not suffix.isdigit():
                    logger.debug(
                        f"Skipping {png_path.name}: expected 9-digit suffix, got '{suffix}'"
                    )
                    continue

                frame_index = int(suffix)

                json_path = png_path.with_suffix(".json")
                if not json_path.exists():
                    logger.debug(
                        f"Skipping {png_path.name}: missing JSON {json_path.name}"
                    )
                    continue
                if not CutieCoreVideoProcessor._json_has_manual_seed_content(json_path):
                    logger.debug(
                        f"Skipping {png_path.name}: JSON does not look like a manual seed."
                    )
                    continue

                existing = collected.get(frame_index)
                if (
                    existing
                    and existing.png_path.stat().st_mtime >= png_path.stat().st_mtime
                ):
                    logger.debug(
                        f"Skipping {png_path.name}: older than registered seed for frame {frame_index}"
                    )
                    continue

                collected[frame_index] = SeedFrame(
                    frame_index=frame_index,
                    png_path=png_path,
                    json_path=json_path,
                )
                logger.info(
                    f"Registered CUTIE seed {png_path.name} (frame {frame_index})"
                )

            return collected

        logger.info(f"Scanning for CUTIE seeds in {results_dir}")
        seeds = collect_seeds(results_dir)

        nested_dir = results_dir / stem
        if not seeds and nested_dir.exists():
            logger.info(
                f"No seeds found at root; scanning nested directory {nested_dir}"
            )
            seeds = collect_seeds(nested_dir)

        discovered = [seeds[idx] for idx in sorted(seeds.keys())]
        CutieCoreVideoProcessor._DISCOVERED_SEEDS_CACHE[cache_key] = list(discovered)
        logger.info(f"Discovered {len(discovered)} CUTIE seed(s) in {results_dir}")
        return discovered

    def _collect_labeled_frame_indices(self) -> Set[int]:
        """Return cached frame indices that already have polygon annotations saved."""
        if self._cached_labeled_frames is not None:
            return self._cached_labeled_frames

        labeled_frames: Set[int] = set()
        results_dir = self.video_folder
        if not results_dir.exists() or not results_dir.is_dir():
            self._cached_labeled_frames = labeled_frames
            return labeled_frames

        stem = results_dir.name
        stem_prefix = f"{stem}_"
        stem_prefix_lower = stem_prefix.lower()

        def has_valid_shapes(shapes: Iterable[Dict[str, Any]]) -> bool:
            for shape in shapes or []:
                if (
                    shape.get("shape_type") == "polygon"
                    and len(shape.get("points", [])) >= 3
                ) or shape.get("mask"):
                    return True
            return False

        # Prefer the annotation store (written during prediction) because per-frame
        # JSON files are not persisted for auto-labeled frames.
        store_path = results_dir / f"{results_dir.name}{AnnotationStore.STORE_SUFFIX}"
        if store_path.exists():
            try:
                store = AnnotationStore(store_path)
                records = store._load_records()
                for frame_idx, record in records.items():
                    try:
                        normalized_idx = int(frame_idx)
                    except (TypeError, ValueError):
                        continue
                    if has_valid_shapes(record.get("shapes")):
                        labeled_frames.add(normalized_idx)
            except Exception as exc:
                logger.debug("Failed to read annotation store %s: %s", store_path, exc)

        def scan_directory(directory: Path):
            if not directory.exists() or not directory.is_dir():
                return
            for json_path in directory.glob("*.json"):
                name_lower = json_path.stem.lower()
                if not name_lower.startswith(stem_prefix_lower):
                    continue
                suffix = name_lower[len(stem_prefix_lower) :]
                if len(suffix) != 9 or not suffix.isdigit():
                    continue
                frame_idx = int(suffix)
                # Skip JSON files for frames already accounted for via the store.
                if frame_idx in labeled_frames:
                    continue
                try:
                    with open(json_path, "r", encoding="utf-8") as fp:
                        data = json.load(fp) or {}
                except (OSError, ValueError) as exc:
                    logger.debug(
                        f"Failed to parse JSON annotation {json_path.name}: {exc}"
                    )
                    continue

                shapes = data.get("shapes") or []
                if not has_valid_shapes(shapes):
                    continue

                labeled_frames.add(frame_idx)

        scan_directory(results_dir)
        nested_dir = results_dir / stem
        if nested_dir.exists():
            scan_directory(nested_dir)

        self._cached_labeled_frames = labeled_frames
        return labeled_frames

    def _apply_seed_cache(self, cache: Dict[str, Any]) -> None:
        """Populate instance seed metadata from a cached entry."""
        self._current_video_cache_key = cache["video_key"]
        self._seed_segment_lookup = cache["segments"]
        self._seed_object_counts = cache["object_counts"]
        self._global_object_ids = list(cache["object_ids"])
        self._global_label_names = dict(cache["label_names"])

    def _prepare_seed_segments(self, seeds: List[SeedFrame]) -> None:
        """Parse every seed frame for the current video and cache the results."""
        video_key = str(Path(self.video_folder).resolve())
        segments: Dict[int, SeedSegment] = {}
        object_counts: Dict[int, int] = {}
        global_label_names: Dict[int, str] = {}
        global_object_ids: Set[int] = set()
        frame_indices: List[int] = []

        for seed in seeds:
            segment = self._load_seed_mask(seed)
            if segment is None:
                continue

            segments[seed.frame_index] = segment
            object_counts[seed.frame_index] = len(segment.active_labels)
            frame_indices.append(seed.frame_index)

            for label in segment.active_labels:
                value = segment.labels_map.get(label)
                if value is None or int(value) == 0:
                    continue
                normalized_value = int(value)
                global_label_names.setdefault(normalized_value, label)
                global_object_ids.add(normalized_value)

        cache_entry = {
            "video_key": video_key,
            "segments": segments,
            "object_counts": object_counts,
            "object_ids": sorted(global_object_ids),
            "label_names": global_label_names,
            "frame_indices": sorted(frame_indices),
        }
        self._video_seed_cache[video_key] = cache_entry
        self._apply_seed_cache(cache_entry)
        logger.info(
            "Cached %d seed frame(s) for video '%s'; discovered %d object id(s).",
            len(segments),
            self.video_name,
            len(global_object_ids),
        )

    def _ensure_seed_cache(self, seeds: List[SeedFrame]) -> None:
        """Ensure the active video has an up-to-date seed cache."""
        video_key = str(Path(self.video_folder).resolve())
        required_frames = sorted(seed.frame_index for seed in seeds)
        cache_entry = self._video_seed_cache.get(video_key)
        cache_frames = cache_entry.get("frame_indices") if cache_entry else None

        if (
            cache_entry is None
            or cache_frames is None
            or not set(required_frames).issubset(cache_frames)
            # detect removed seeds
            or set(cache_frames) != set(required_frames)
        ):
            self._prepare_seed_segments(seeds)
        else:
            self._apply_seed_cache(cache_entry)

    def _register_active_objects(self, object_ids: Iterable[int]) -> None:
        """Track which object ids have appeared so far in the current video."""
        updated = False
        for obj_id in object_ids or []:
            if obj_id is None:
                continue
            normalized = int(obj_id)
            if normalized not in self._video_active_object_ids:
                self._video_active_object_ids.add(normalized)
                updated = True
        if updated or not self.num_tracking_instances:
            self.num_tracking_instances = max(
                self.num_tracking_instances,
                len(self._video_active_object_ids),
            )

    @staticmethod
    def _build_frame_intervals(frame_indices: Set[int]) -> List[Tuple[int, int]]:
        """Convert sparse frame indices into sorted closed intervals."""
        if not frame_indices:
            return []
        sorted_indices = sorted(int(idx) for idx in frame_indices)
        intervals: List[Tuple[int, int]] = []
        start = sorted_indices[0]
        end = start
        for idx in sorted_indices[1:]:
            if idx == end + 1:
                end = idx
                continue
            intervals.append((start, end))
            start = idx
            end = idx
        intervals.append((start, end))
        return intervals

    @staticmethod
    def _range_fully_covered(
        intervals: List[Tuple[int, int]], start: int, end: int
    ) -> bool:
        """Return True if [start, end] is fully covered by interval union."""
        if start > end:
            return True
        if not intervals:
            return False
        starts = [it[0] for it in intervals]
        idx = bisect_right(starts, start) - 1
        if idx < 0:
            return False
        cur_start, cur_end = intervals[idx]
        if cur_start > start or cur_end < start:
            return False
        target = start
        while True:
            if cur_end >= end:
                return True
            target = cur_end + 1
            idx += 1
            if idx >= len(intervals):
                return False
            nxt_start, nxt_end = intervals[idx]
            if nxt_start > target:
                return False
            cur_end = max(cur_end, nxt_end)

    @staticmethod
    def _segment_already_completed(
        segment: SeedSegment,
        resolved_end: int,
        labeled_frames: Set[int],
        labeled_intervals: Optional[List[Tuple[int, int]]] = None,
    ) -> bool:
        """Return True if every frame in the segment range already has annotations."""
        if resolved_end is None:
            return False
        if resolved_end < segment.start_frame:
            return True
        intervals = (
            labeled_intervals
            if labeled_intervals is not None
            else CutieCoreVideoProcessor._build_frame_intervals(labeled_frames)
        )
        return CutieCoreVideoProcessor._range_fully_covered(
            intervals, int(segment.start_frame), int(resolved_end)
        )

    def initialize_video_writer(
        self, output_video_path, frame_width, frame_height, fps=30
    ):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            output_video_path, fourcc, fps, (frame_width, frame_height)
        )

    def _initialize_model(self):
        # general setup
        torch.cuda.empty_cache()
        with torch.inference_mode():
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            initialize(
                version_base="1.3.2", config_path="config", job_name="eval_config"
            )
            cfg = compose(config_name="eval_config")
            model_path = os.path.join(
                self.current_folder, "weights/cutie-base-mega.pth"
            )
            if not os.path.exists(model_path):
                gdown.cached_download(self._REMOTE_MODEL_URL, model_path, md5=self._MD5)
            with open_dict(cfg):
                cfg["weights"] = model_path
                cfg["max_mem_frames"] = self.max_mem_frames
            cfg["mem_every"] = self.mem_every
            logger.info(f"Saving into working memory for every: {self.mem_every}.")
            logger.info(f"Tmax: max_mem_frames: {self.max_mem_frames}")
            cutie_model = CUTIE(cfg).to(self.device).eval()
            model_weights = torch.load(
                cfg.weights, map_location=self.device, weights_only=True
            )
            cutie_model.load_weights(model_weights)
        return cutie_model, cfg

    def _save_bbox(self, points, frame_area, label):
        # A linearring requires at least 4 coordinates.
        # good quality polygon
        if len(points) >= 4:
            xs = [float(pt[0]) for pt in points]
            ys = [float(pt[1]) for pt in points]
            minx = min(xs)
            miny = min(ys)
            maxx = max(xs)
            maxy = max(ys)
            _bbox = (minx, miny, maxx, maxy)
            # Calculate the area of the bounding box
            bbox_area = (_bbox[2] - _bbox[0]) * (_bbox[3] - _bbox[1])
            # bbox area should bigger enough
            if bbox_area <= (frame_area * 0.50) and bbox_area >= (frame_area * 0.0002):
                self.cache.add_bbox(label, _bbox)

    def _save_results(self, label, mask):
        try:
            cx, cy = find_mask_center_opencv(mask)
        except ZeroDivisionError as e:
            logger.info(e)
            return None
        cx = self._normalize_tracking_scalar(cx, default=0.0)
        cy = self._normalize_tracking_scalar(cy, default=0.0)
        if self._flow_hsv is not None:
            # unnormalized magnitude
            magnitude = self._flow_hsv[..., 2]
            magnitude = magnitude.astype(np.float32)
            mask_sum = np.sum(mask)
            if mask_sum > 0:
                self._motion_index = np.sum(mask * magnitude) / mask_sum
            else:
                self._motion_index = 0.0
        else:
            self._motion_index = -1
        self._motion_index = self._normalize_tracking_scalar(
            self._motion_index, default=-1.0
        )
        return cx, cy, self._motion_index

    def _save_annotation(self, filename, mask_dict, frame_shape):
        return self._save_annotation_with_notes(
            filename=filename,
            mask_dict=mask_dict,
            frame_shape=frame_shape,
            shape_notes=None,
        )

    def _save_annotation_with_notes(
        self,
        filename,
        mask_dict,
        frame_shape,
        shape_notes: Optional[Dict[str, str]] = None,
    ):
        height, width, _ = frame_shape
        frame_area = height * width
        label_list = []
        persisted_masks: Dict[str, np.ndarray] = {}
        shape_notes = shape_notes or {}
        for label_id, mask in mask_dict.items():
            label = str(label_id)
            mask, corrected = self._sanitize_full_frame_artifact(
                label, mask, float(frame_area)
            )
            if corrected:
                self._log_repetitive_warning(
                    ("full_frame_corrected", label),
                    "CUTIE full-frame mask corrected for '%s' at frame %s."
                    % (label, self._frame_number),
                )
            if self.reject_suspicious_mask_jumps and self._is_suspicious_mask_jump(
                label, mask, float(frame_area)
            ):
                self._log_repetitive_warning(
                    ("mask_jump_rejected", label),
                    "CUTIE mask jump rejected for '%s' at frame %s (possible full-frame artifact)."
                    % (label, self._frame_number),
                )
                continue

            metrics = self._save_results(label, mask)
            if metrics is None:
                continue
            cx, cy, motion_index = metrics
            self.save_KMedoids_in_mask(label_list, mask)
            note_text = str(shape_notes.get(label, "") or "").strip()
            description = f"motion_index: {motion_index}"
            if note_text:
                description += f"; note: {note_text}"

            current_shape = MaskShape(
                label=label,
                flags={},
                description=description,
            )
            current_shape.other_data = {
                "cx": cx,
                "cy": cy,
                "motion_index": motion_index,
            }
            if note_text:
                current_shape.other_data["note"] = note_text
            current_shape.mask = mask
            _shapes = current_shape.toPolygons(epsilon=self.epsilon_for_polygon)
            if len(_shapes) <= 0:
                continue
            current_shape = _shapes[0]
            points = [[point.x(), point.y()] for point in current_shape.points]
            if self._should_reject_frame_sized_prediction(
                label=label,
                mask=mask,
                points=points,
                frame_area=float(frame_area),
            ):
                fallback_mask = self._recent_instance_masks.get(label)
                if fallback_mask is not None:
                    fallback_mask = np.asarray(fallback_mask).astype(bool)
                if fallback_mask is None or fallback_mask.ndim != 2 or not fallback_mask.any():
                    self._log_repetitive_warning(
                        ("frame_sized_rejected", label),
                        "CUTIE frame-sized artifact rejected for '%s' at frame %s."
                        % (label, self._frame_number),
                    )
                    continue

                self._log_repetitive_warning(
                    ("frame_sized_fallback", label),
                    "CUTIE frame-sized artifact rejected for '%s' at frame %s; "
                    "falling back to last good mask."
                    % (label, self._frame_number),
                )
                mask = fallback_mask.copy()
                note_text = str(shape_notes.get(label, "") or "").strip()
                fallback_note = "fallback_previous_mask_after_frame_sized_artifact"
                if note_text:
                    note_text = f"{note_text}; {fallback_note}"
                else:
                    note_text = fallback_note
                description = f"motion_index: {motion_index}; note: {note_text}"
                current_shape = MaskShape(
                    label=label,
                    flags={},
                    description=description,
                )
                current_shape.other_data = {
                    "cx": cx,
                    "cy": cy,
                    "motion_index": motion_index,
                    "note": note_text,
                }
                current_shape.mask = mask
                _shapes = current_shape.toPolygons(epsilon=self.epsilon_for_polygon)
                if len(_shapes) <= 0:
                    continue
                current_shape = _shapes[0]
                points = [[point.x(), point.y()] for point in current_shape.points]
                if self._should_reject_frame_sized_prediction(
                    label=label,
                    mask=mask,
                    points=points,
                    frame_area=float(frame_area),
                ):
                    continue
            self._save_bbox(points, frame_area, label)
            current_shape.points = points
            label_list.append(current_shape)
            persisted_masks[label] = np.asarray(mask).astype(bool)
            self._last_mask_area_ratio[label] = float(np.count_nonzero(mask)) / max(
                float(frame_area), 1.0
            )
        self._last_saved_instance_masks = persisted_masks
        save_labels(
            filename=filename,
            imagePath=None,
            label_list=label_list,
            height=height,
            width=width,
            save_image_to_json=False,
            persist_json=False,
        )
        return label_list

    def _sanitize_full_frame_artifact(
        self, label: str, mask: np.ndarray, frame_area: float
    ) -> tuple[np.ndarray, bool]:
        """Keep component near cached bbox when mask expands to near full frame."""
        if frame_area <= 0:
            return mask, False
        current_area = float(np.count_nonzero(mask))
        if current_area <= 0:
            return mask, False
        if (current_area / frame_area) < 0.98:
            return mask, False

        recent_bbox = self.cache.get_most_recent_bbox(label)
        if recent_bbox is None:
            return mask, False
        x1, y1, x2, y2 = recent_bbox
        bbox_area = max(0.0, float(x2 - x1) * float(y2 - y1))
        bbox_ratio = bbox_area / frame_area
        if bbox_ratio <= 0.0 or bbox_ratio >= 0.85:
            return mask, False

        mask_u8 = mask.astype(np.uint8)
        num_labels, components, stats, _ = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8
        )
        if num_labels <= 1:
            return mask, False

        h, w = mask.shape[:2]
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        pad_x = max(8, int(bw))
        pad_y = max(8, int(bh))
        rx1 = max(0, int(np.floor(x1)) - pad_x)
        ry1 = max(0, int(np.floor(y1)) - pad_y)
        rx2 = min(w, int(np.ceil(x2)) + pad_x)
        ry2 = min(h, int(np.ceil(y2)) + pad_y)
        if rx2 <= rx1 or ry2 <= ry1:
            return mask, False

        best_id = 0
        best_overlap = 0
        best_area = 0
        for comp_id in range(1, num_labels):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            overlap = int(np.count_nonzero(components[ry1:ry2, rx1:rx2] == comp_id))
            if overlap <= 0:
                continue
            if overlap > best_overlap or (overlap == best_overlap and area > best_area):
                best_id = comp_id
                best_overlap = overlap
                best_area = area

        if best_id == 0:
            return mask, False

        cleaned = components == best_id
        cleaned_area = float(np.count_nonzero(cleaned))
        if cleaned_area <= 0:
            return mask, False
        cleaned_ratio = cleaned_area / frame_area
        current_ratio = current_area / frame_area
        if cleaned_ratio >= current_ratio or cleaned_ratio >= 0.98:
            return mask, False
        return cleaned, True

    def _should_reject_frame_sized_prediction(
        self, label: str, mask: np.ndarray, points: List[List[float]], frame_area: float
    ) -> bool:
        """Reject near-full-frame masks when prior bbox/area history indicates artifact."""
        if frame_area <= 0:
            return False
        current_area = float(np.count_nonzero(mask))
        if current_area <= 0:
            return False
        current_ratio = current_area / frame_area
        if current_ratio < 0.97:
            return False

        touches_all_borders = bool(
            mask[0, :].any()
            and mask[-1, :].any()
            and mask[:, 0].any()
            and mask[:, -1].any()
        )
        if not touches_all_borders:
            return False

        if not points:
            return False
        xs = [float(pt[0]) for pt in points]
        ys = [float(pt[1]) for pt in points]
        bbox_area = max(0.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
        bbox_ratio = bbox_area / frame_area
        if bbox_ratio < 0.94:
            return False

        recent_bbox = self.cache.get_most_recent_bbox(label)
        if recent_bbox is not None:
            x1, y1, x2, y2 = recent_bbox
            recent_bbox_area = max(0.0, float(x2 - x1) * float(y2 - y1))
            recent_ratio = recent_bbox_area / frame_area
            if 0.0 < recent_ratio < 0.75:
                return True

        previous_ratio = self._last_mask_area_ratio.get(label)
        if previous_ratio is None:
            return False
        if previous_ratio >= 0.75:
            return False

        growth_ratio = current_ratio / max(previous_ratio, 1.0 / frame_area)
        return growth_ratio >= 1.35

    def _is_suspicious_mask_jump(
        self, label: str, mask: np.ndarray, frame_area: float
    ) -> bool:
        if frame_area <= 0:
            return False
        current_area = float(np.count_nonzero(mask))
        if current_area <= 0:
            return False
        current_ratio = current_area / frame_area
        if current_ratio < 0.998:
            return False

        # Only treat as suspicious when the mask spans the full frame borders.
        touches_all_borders = bool(
            mask[0, :].any()
            and mask[-1, :].any()
            and mask[:, 0].any()
            and mask[:, -1].any()
        )
        if not touches_all_borders:
            return False

        previous_ratio = self._last_mask_area_ratio.get(label)
        if previous_ratio is None:
            return current_ratio >= 0.9995

        # If the object was already large, do not block it.
        if previous_ratio >= 0.80:
            return False

        growth_ratio = current_ratio / max(previous_ratio, 1.0 / frame_area)
        if growth_ratio < 1.4:
            return False

        recent_bbox = self.cache.get_most_recent_bbox(label)
        if recent_bbox is not None:
            x1, y1, x2, y2 = recent_bbox
            bbox_area = max(0.0, float(x2 - x1) * float(y2 - y1))
            bbox_ratio = bbox_area / frame_area
            if bbox_ratio > 0.60:
                return False

        return True

    def save_KMedoids_in_mask(self, label_list, mask):
        if self._flow is not None and self.showing_KMedoids_in_mask:
            flow_points = extract_flow_points_in_mask(mask, self._flow)
            for fpoint in flow_points.tolist():
                fpoint_shape = Shape(
                    label="kmedoids",
                    flags={},
                    shape_type="point",
                    description="kmedoids of flow in mask",
                )
                fpoint_shape.points = [fpoint]
                label_list.append(fpoint_shape)

    def _recover_missing_instances_with_bbox(
        self, instance_names, cur_frame, score_threshold=0.60
    ) -> Tuple[Dict[str, np.ndarray], Set[str]]:
        """Recover missing masks from cached bboxes in a single SAM call.

        Returns:
            recovered_masks: recovered label->mask mappings.
            attempted_labels: labels that were actually sent to SAM for recovery.
        """
        if not instance_names:
            return {}, set()
        if self.sam_hq is None or not hasattr(self.sam_hq, "segment_objects"):
            logger.debug("Skipping missing-instance recovery: SAM HQ is unavailable.")
            return {}, set()

        valid_names: List[str] = []
        valid_boxes: List[Tuple[float, float, float, float]] = []
        for instance_name in sorted(instance_names):
            bbox = self.cache.get_most_recent_bbox(instance_name)
            if bbox is None:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in bbox)
            except Exception:
                continue
            if not np.isfinite([x1, y1, x2, y2]).all():
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            valid_names.append(str(instance_name))
            valid_boxes.append((x1, y1, x2, y2))

        if not valid_boxes:
            return {}, set()

        attempted_labels = set(valid_names)
        try:
            masks, scores, _ = self.sam_hq.segment_objects(cur_frame, valid_boxes)
        except Exception as exc:
            logger.warning(
                "Missing-instance recovery failed for frame %s: %s",
                self._frame_number,
                exc,
            )
            return {}, attempted_labels

        recovered: Dict[str, np.ndarray] = {}
        num_items = min(len(valid_names), len(masks), len(scores))
        for idx in range(num_items):
            instance_name = valid_names[idx]
            score = self._normalize_tracking_scalar(scores[idx], default=0.0)
            logger.info(
                "BBox recovery candidate %s score=%.4f threshold=%.2f",
                instance_name,
                score,
                score_threshold,
            )
            if score < score_threshold:
                continue

            mask_arr = np.asarray(masks[idx])
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[0]
            if mask_arr.ndim != 2:
                continue
            mask_bool = mask_arr.astype(bool)
            if not mask_bool.any():
                continue
            recovered[instance_name] = mask_bool

        return recovered, attempted_labels

    # Backward-compatible alias (legacy misspelling).
    def segement_with_bbox(self, instance_names, cur_frame, score_threshold=0.60):
        recovered, _ = self._recover_missing_instances_with_bbox(
            instance_names, cur_frame, score_threshold=score_threshold
        )
        return recovered

    def _update_recent_instance_masks(self, frame_idx: int, mask_dict: Dict[str, np.ndarray]) -> None:
        """Store latest available binary mask per instance for fallback fill."""
        for label, mask in (mask_dict or {}).items():
            try:
                mask_bool = np.asarray(mask).astype(bool)
            except Exception:
                continue
            if mask_bool.ndim != 2 or not mask_bool.any():
                continue
            key = str(label)
            self._recent_instance_masks[key] = mask_bool.copy()
            self._recent_instance_mask_frames[key] = int(frame_idx)

    def _fill_missing_instances_from_recent_masks(
        self, missing_instances: Set[str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        """Fill missing instances using their most recent available mask."""
        recovered: Dict[str, np.ndarray] = {}
        notes: Dict[str, str] = {}
        for instance_name in sorted(missing_instances):
            key = str(instance_name)
            prev_mask = self._recent_instance_masks.get(key)
            if prev_mask is None:
                continue
            prev_frame = self._recent_instance_mask_frames.get(key)
            recovered[key] = prev_mask.copy()
            if prev_frame is None:
                notes[key] = "filled_from_previous_available_instance_mask"
            else:
                notes[key] = f"filled_from_previous_available_instance_mask(frame={prev_frame})"
        return recovered, notes

    def shapes_to_mask(self, label_json_file, image_size):
        """
        Convert label JSON file containing shapes to a binary mask.

        Args:
            label_json_file (str): Path to the label JSON file.
            image_size (tuple): Size of the image.

        Returns:
            tuple: Tuple containing the binary mask and the
            dictionary mapping label names to their values.
        """
        label_name_to_value = {"_background_": 0}
        with open(label_json_file, "r") as json_file:
            data = json.load(json_file)

        filtered_shapes = []
        for shape in data.get("shapes", []):
            points = shape.get("points") or []
            if len(points) < 3:
                continue

            label_text = (shape.get("label") or "").lower()
            description_text = (shape.get("description") or "").lower()
            if "zone" in label_text or "zone" in description_text:
                continue

            flags = shape.get("flags") or {}
            if any(
                str(flag_key).lower() == "zone" and bool(flag_val)
                for flag_key, flag_val in flags.items()
            ):
                continue

            filtered_shapes.append(shape)

        for shape in sorted(filtered_shapes, key=lambda x: x.get("label") or ""):
            label_name = shape.get("label")
            if not label_name:
                continue
            if label_name not in label_name_to_value:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value

        if len(label_name_to_value) <= 1:
            return None, label_name_to_value

        mask, _ = shapes_to_label(image_size, filtered_shapes, label_name_to_value)
        return mask, label_name_to_value

    def _assign_global_ids(self, label_name_to_value: Dict[str, int]) -> Dict[str, int]:
        assigned: Dict[str, int] = {"_background_": 0}
        for label, _ in sorted(label_name_to_value.items(), key=lambda item: item[1]):
            if label == "_background_":
                continue
            if label not in self.label_registry:
                self.label_registry[label] = len(self.label_registry)
            assigned[label] = self.label_registry[label]
        return assigned

    @staticmethod
    def _remap_mask_to_global(
        mask: np.ndarray,
        original_label_map: Dict[str, int],
        global_label_map: Dict[str, int],
    ) -> np.ndarray:
        remapped = np.zeros_like(mask, dtype=np.int32)
        for label, original_value in original_label_map.items():
            global_value = global_label_map.get(label, 0)
            remapped[mask == original_value] = global_value
        return remapped

    @staticmethod
    def _extract_active_labels(
        mask: np.ndarray, global_label_map: Dict[str, int]
    ) -> List[str]:
        active: List[str] = []
        for label, value in global_label_map.items():
            if label == "_background_":
                continue
            if np.any(mask == value):
                active.append(label)
        return active

    def _build_object_mask_tensor(self, mask: np.ndarray):
        unique_ids = sorted(int(v) for v in np.unique(mask) if v != 0)
        if not unique_ids:
            return None, []
        num_classes = int(mask.max()) + 1
        one_hot = index_numpy_to_one_hot_torch(mask.astype(np.int64), num_classes)
        selected = one_hot[unique_ids]
        return selected, unique_ids

    def _build_seed_mask_dict(
        self,
        segment: SeedSegment,
        expected_labels: Set[str],
    ) -> Dict[str, np.ndarray]:
        """Build per-label masks directly from the seed segment."""
        seed_masks: Dict[str, np.ndarray] = {}
        for label in sorted(expected_labels):
            value = segment.labels_map.get(label)
            if value is None:
                continue
            label_mask = segment.mask == int(value)
            if not np.any(label_mask):
                continue
            seed_masks[str(label)] = label_mask
        return seed_masks

    @staticmethod
    def _map_local_prediction_to_global(
        prediction: np.ndarray, active_global_ids: List[int]
    ) -> np.ndarray:
        global_mask = np.zeros_like(prediction, dtype=np.int32)
        for local_idx, global_id in enumerate(active_global_ids, start=1):
            global_mask[prediction == local_idx] = global_id
        return global_mask

    def _load_seed_mask(self, seed: SeedFrame) -> Optional[SeedSegment]:
        """Load mask and label mapping for a given seed frame."""
        frame = cv2.imread(str(seed.png_path))
        if frame is None:
            logger.warning(f"Failed to read seed image: {seed.png_path}")
            return None

        mask, label_map = self.shapes_to_mask(str(seed.json_path), frame.shape[:2])
        if mask is None or len(np.unique(mask)) <= 1:
            logger.warning(
                f"Seed {seed.png_path.name} has no valid polygon annotations; skipping."
            )
            return None

        if "_background_" not in label_map:
            label_map = {"_background_": 0, **label_map}
        global_label_map = self._assign_global_ids(label_map)
        remapped_mask = self._remap_mask_to_global(mask, label_map, global_label_map)
        active_labels = self._extract_active_labels(remapped_mask, global_label_map)

        segment = SeedSegment(
            seed=seed,
            start_frame=seed.frame_index,
            end_frame=None,
            mask=remapped_mask,
            labels_map=global_label_map,
            active_labels=active_labels,
        )
        self._seed_bbox_cache_from_segment(segment)
        return segment

    def _seed_bbox_cache_from_segment(self, segment: SeedSegment) -> None:
        """Prime bbox cache from seed masks so early-frame correction has context."""
        try:
            frame_area = float(segment.mask.shape[0] * segment.mask.shape[1])
            if frame_area <= 0:
                return
            for label in segment.active_labels:
                value = segment.labels_map.get(label)
                if value is None or int(value) == 0:
                    continue
                ys, xs = np.where(segment.mask == int(value))
                if xs.size == 0 or ys.size == 0:
                    continue
                x1 = float(xs.min())
                y1 = float(ys.min())
                x2 = float(xs.max())
                y2 = float(ys.max())
                bbox_area = max(0.0, (x2 - x1) * (y2 - y1))
                if bbox_area <= 0.0 or bbox_area >= frame_area * 0.95:
                    continue
                self.cache.add_bbox(label, (x1, y1, x2, y2))
        except Exception:
            logger.debug("Failed to seed CUTIE bbox cache.", exc_info=True)

    def _build_seed_segments(
        self, seeds: List[SeedFrame], requested_end: Optional[int]
    ) -> List[SeedSegment]:
        """Create contiguous segments from discovered seeds."""
        segments: List[SeedSegment] = []
        ordered_unique_seeds = sorted(
            {int(seed.frame_index): seed for seed in seeds}.values(),
            key=lambda seed: int(seed.frame_index),
        )
        for idx, seed in enumerate(ordered_unique_seeds):
            cached_segment = self._seed_segment_lookup.get(seed.frame_index)
            if cached_segment is None:
                cached_segment = self._load_seed_mask(seed)
                if cached_segment is None:
                    continue
                self._seed_segment_lookup[seed.frame_index] = cached_segment
                self._seed_object_counts[seed.frame_index] = len(
                    cached_segment.active_labels
                )
                for label in cached_segment.active_labels:
                    value = cached_segment.labels_map.get(label)
                    if value is None or int(value) == 0:
                        continue
                    normalized_value = int(value)
                    if normalized_value not in self._global_label_names:
                        self._global_label_names[normalized_value] = label
                        self._global_object_ids.append(normalized_value)
                self._global_object_ids.sort()

            segment = replace(cached_segment)
            segment.start_frame = seed.frame_index
            segment.end_frame = None

            next_seed_frame = None
            if idx + 1 < len(ordered_unique_seeds):
                next_seed_frame = ordered_unique_seeds[idx + 1].frame_index

            if requested_end is not None:
                segment.end_frame = requested_end
            if next_seed_frame is not None:
                candidate_end = next_seed_frame - 1
                if segment.end_frame is None:
                    segment.end_frame = candidate_end
                else:
                    segment.end_frame = min(segment.end_frame, candidate_end)

            if (
                segment.end_frame is not None
                and segment.end_frame < segment.start_frame
            ):
                logger.info(
                    f"Skipping degenerate seed range [{segment.start_frame}, {segment.end_frame}]."
                )
                continue

            segments.append(segment)

        if requested_end is not None and segments:
            # Ensure the final segment honours requested_end.
            segments[-1].end_frame = min(
                requested_end,
                segments[-1].end_frame
                if segments[-1].end_frame is not None
                else requested_end,
            )

        return segments

    def commit_masks_into_permanent_memory(
        self,
        frame_number,
        labels_dict,
        seed_frames: Optional[List[SeedFrame]] = None,
        seed_segment_lookup: Optional[Dict[int, SeedSegment]] = None,
        segment_end: Optional[int] = None,
    ):
        """
        Commit masks into permanent memory for inference.

        Args:
            frame_number (int): Frame number.
            labels_dict (dict): Dictionary mapping label names to their values.

        Returns:
            dict: Updated labels dictionary.
        """
        with torch.inference_mode():
            with torch.amp.autocast(
                "cuda", enabled=self.cfg.amp and self.device == "cuda"
            ):
                candidate_seeds = seed_frames or self._seed_frames
                if not candidate_seeds:
                    candidate_seeds = self.discover_seed_frames(
                        self.video_name, self.video_folder
                    )

                for seed in candidate_seeds:
                    if seed.frame_index <= frame_number:
                        continue
                    if seed.frame_index in self._committed_seed_frames:
                        continue
                    if segment_end is not None and seed.frame_index > segment_end:
                        continue

                    segment = None
                    if seed_segment_lookup:
                        segment = seed_segment_lookup.get(seed.frame_index)
                    if segment is None:
                        segment = self._seed_segment_lookup.get(seed.frame_index)
                    if segment is None:
                        segment = self._load_seed_mask(seed)
                        if segment and segment.seed is not None:
                            self._seed_segment_lookup[segment.seed.frame_index] = (
                                segment
                            )
                    if segment is None:
                        continue

                    if not segment.active_labels:
                        continue

                    frame = (
                        cv2.imread(str(segment.seed.png_path)) if segment.seed else None
                    )
                    if frame is None:
                        logger.warning(
                            f"Failed to read seed frame for committing permanent memory: {segment.seed}"
                        )
                        continue

                    for label in segment.active_labels:
                        labels_dict.setdefault(label, segment.labels_map[label])

                    frame_torch = image_to_torch(frame, device=self.device)
                    mask_tensor, active_ids = self._build_object_mask_tensor(
                        segment.mask
                    )
                    if mask_tensor is None or not active_ids:
                        continue
                    self._register_active_objects(active_ids)
                    mask_tensor = mask_tensor.to(self.device)
                    self.processor.step(
                        frame_torch,
                        mask_tensor,
                        objects=active_ids,
                        idx_mask=False,
                        force_permanent=True,
                    )
                    self._committed_seed_frames.add(segment.seed.frame_index)
                    logger.info(
                        f"Committed {len(active_ids)} instances from seed #{segment.seed.frame_index} into permanent memory."
                    )

                return labels_dict

    def _run_segments(
        self,
        segments: List[SeedSegment],
        pred_worker=None,
        recording: bool = False,
        output_video_path: Optional[str] = None,
        has_occlusion: bool = False,
        seed_frames: Optional[List[SeedFrame]] = None,
        seed_segment_lookup: Optional[Dict[int, SeedSegment]] = None,
        visualize_every: int = 30,
    ) -> str:
        if not segments:
            return "No valid segments found for CUTIE processing."
        self._last_mask_area_ratio = {}

        # Refresh cached labeled frames so resume logic sees the latest edits
        self._cached_labeled_frames = None
        labeled_frames = self._collect_labeled_frame_indices()
        labeled_intervals = self._build_frame_intervals(labeled_frames)

        cap = cv2.VideoCapture(self.video_name)
        if not cap.isOpened():
            if pred_worker is not None:
                pred_worker.stop_signal.emit()
            return f"Failed to open video {self.video_name}"

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last_frame_index = max(0, total_frames - 1)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if recording:
            if output_video_path is None:
                output_video_path = self.output_tracking_csvpath.replace(".csv", ".mp4")
            self.initialize_video_writer(
                output_video_path, frame_width, frame_height, fps
            )
            logger.info(f"Saving the color masks to video {output_video_path}")

        final_message: Optional[str] = None
        halt_requested = False
        processed_any_segment = False
        skipped_segments = 0
        max_resolved_end = -1

        if seed_segment_lookup is None:
            seed_segment_lookup = {
                segment.seed.frame_index: segment
                for segment in segments
                if segment.seed is not None
            }

        try:
            for idx, segment in enumerate(segments):
                if self._should_stop(pred_worker):
                    halt_requested = True
                    break

                resolved_end = segment.end_frame
                if resolved_end is None:
                    resolved_end = last_frame_index
                else:
                    resolved_end = min(resolved_end, last_frame_index)

                if resolved_end < segment.start_frame:
                    logger.info(
                        f"Skipping segment starting at {segment.start_frame} with end {resolved_end}."
                    )
                    continue
                max_resolved_end = max(max_resolved_end, resolved_end)

                if self._segment_already_completed(
                    segment, resolved_end, labeled_frames, labeled_intervals
                ):
                    skipped_segments += 1
                    logger.info(
                        f"Skipping Cutie segment [{segment.start_frame}, {resolved_end}] - annotations already exist."
                    )
                    continue

                message, should_halt = self._process_segment(
                    cap=cap,
                    segment=segment,
                    end_frame=resolved_end,
                    fps=fps,
                    pred_worker=pred_worker,
                    recording=recording,
                    has_occlusion=has_occlusion,
                    seed_frames=seed_frames,
                    seed_segment_lookup=seed_segment_lookup,
                    visualize_every=visualize_every,
                )

                processed_any_segment = True

                if message:
                    final_message = message

                if should_halt:
                    halt_requested = True
                    break

                # Mark this processed span as complete for fast overlap checks
                # when subsequent seed segments share frame ranges.
                labeled_frames.update(range(segment.start_frame, resolved_end + 1))
                labeled_intervals = self._build_frame_intervals(labeled_frames)

            if (
                not processed_any_segment
                and skipped_segments > 0
                and final_message is None
            ):
                final_message = (
                    "CUTIE processing skipped. All selected segment ranges already "
                    "contained saved annotations."
                )

            if final_message is None:
                last_segment = segments[-1]
                resolved_end = last_segment.end_frame
                if resolved_end is None:
                    resolved_end = last_frame_index
                final_message = (
                    "Stop at frame:\n"
                    + f"#{max(last_segment.start_frame, resolved_end)}"
                )
        finally:
            cap.release()
            if recording and hasattr(self, "video_writer"):
                self.video_writer.release()

        reached_video_end = max_resolved_end >= last_frame_index
        if (
            not halt_requested
            and not self._should_stop(pred_worker)
            and reached_video_end
        ):
            self._write_tracking_csv(fps)
        elif not halt_requested and not self._should_stop(pred_worker):
            logger.info(
                "Skipping tracked CSV/gap report generation because CUTIE did not reach video end "
                "(max resolved end=%s, last frame=%s).",
                max_resolved_end,
                last_frame_index,
            )

        if pred_worker is not None and not halt_requested:
            pred_worker.stop_signal.emit()

        return final_message or "CUTIE processing completed."

    def _process_segment(
        self,
        cap,
        segment: SeedSegment,
        end_frame: int,
        fps: float,
        pred_worker=None,
        recording: bool = False,
        has_occlusion: bool = False,
        seed_frames: Optional[List[SeedFrame]] = None,
        seed_segment_lookup: Optional[Dict[int, SeedSegment]] = None,
        visualize_every: int = 30,
    ) -> (Optional[str], bool):
        """Run CUTIE on a single contiguous segment."""

        labels_dict = dict(segment.labels_map)
        self.processor = InferenceCore(self.cutie, cfg=self.cfg)
        try:
            _labels_dict = self.commit_masks_into_permanent_memory(
                segment.start_frame,
                labels_dict,
                seed_frames=seed_frames,
                seed_segment_lookup=seed_segment_lookup,
                segment_end=end_frame,
            )
            labels_dict.update(_labels_dict)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.info(exc)

        mask = segment.mask.astype(np.int32)
        if mask is None:
            return (f"Seed at frame {segment.start_frame} has no valid mask.", True)

        mask_tensor, active_ids = self._build_object_mask_tensor(mask)
        if mask_tensor is None or not active_ids:
            return (
                f"No valid polygon found in seed frame #{segment.start_frame}",
                True,
            )

        mask_tensor = mask_tensor.to(self.device)
        self._register_active_objects(active_ids)
        value_to_label_names = {
            value: label for label, value in self.label_registry.items()
        }
        if self._global_label_names:
            value_to_label_names.update(self._global_label_names)
        instance_names = set(segment.active_labels)
        expected_instance_count = len(instance_names)
        local_to_global = np.zeros(len(active_ids) + 1, dtype=np.int32)
        for local_idx, global_id in enumerate(active_ids, start=1):
            local_to_global[local_idx] = int(global_id)

        current_frame_index = segment.start_frame
        prev_frame = None
        delimiter = "#"
        segment_length = max(1, int(end_frame - segment.start_frame + 1))
        progress_log_every = max(100, min(1000, segment_length // 200 or 1))
        missing_log_every = 25
        last_missing_key: Optional[Tuple[str, ...]] = None
        missing_streak = 0
        suppressed_missing_logs = 0

        logger.info(
            "Processing Cutie segment [%s, %s] (%s frames, %s tracked instance(s)).",
            segment.start_frame,
            end_frame,
            segment_length,
            expected_instance_count,
        )

        if current_frame_index >= end_frame:
            # Still process the single frame
            end_frame = current_frame_index

        with torch.inference_mode():
            with torch.amp.autocast(
                "cuda", enabled=self.cfg.amp and self.device == "cuda"
            ):
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != current_frame_index:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
                while cap.isOpened():
                    if self._should_stop(pred_worker):
                        return (None, True)

                    if current_frame_index > end_frame:
                        break

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    if self._should_stop(pred_worker):
                        return (None, True)

                    self._frame_number = current_frame_index
                    processed_in_segment = (
                        current_frame_index - int(segment.start_frame) + 1
                    )
                    if (
                        processed_in_segment == 1
                        or processed_in_segment % progress_log_every == 0
                    ):
                        pct = int(
                            min(
                                100,
                                max(
                                    0,
                                    round((processed_in_segment / segment_length) * 100),
                                ),
                            )
                        )
                        logger.info(
                            "Cutie segment progress: frame %s/%s (%s%%).",
                            current_frame_index,
                            end_frame,
                            pct,
                        )
                        try:
                            if pred_worker is not None:
                                if hasattr(pred_worker, "report_progress"):
                                    pred_worker.report_progress(pct)
                                elif hasattr(pred_worker, "progress_signal"):
                                    pred_worker.progress_signal.emit(pct)
                        except Exception:
                            pass
                    frame_torch = image_to_torch(frame, device=self.device)
                    filename = self.video_folder / (
                        self.video_folder.name + f"_{current_frame_index:0>{9}}.json"
                    )

                    if current_frame_index == segment.start_frame:
                        try:
                            prediction = self.processor.step(
                                frame_torch,
                                mask_tensor,
                                objects=active_ids,
                                idx_mask=False,
                                force_permanent=True,
                            )
                            prediction = torch_prob_to_numpy_mask(prediction)
                        except Exception as exc:  # pragma: no cover - logging only
                            logger.info(exc)
                            prediction = None
                    else:
                        prediction = self.processor.step(frame_torch)
                        prediction = torch_prob_to_numpy_mask(prediction)

                    if self._should_stop(pred_worker):
                        return (None, True)

                    if prediction is None:
                        global_prediction = mask.copy()
                        mask_dict = {}
                        for global_id in active_ids:
                            global_mask = global_prediction == int(global_id)
                            if not global_mask.any():
                                continue
                            label_name = value_to_label_names.get(
                                int(global_id), str(global_id)
                            )
                            mask_dict[label_name] = global_mask
                    else:
                        mask_dict = {}
                        for local_idx, global_id in enumerate(active_ids, start=1):
                            local_mask = prediction == local_idx
                            if not local_mask.any():
                                continue
                            label_name = value_to_label_names.get(
                                int(global_id), str(global_id)
                            )
                            mask_dict[label_name] = local_mask

                        global_prediction = None
                        if recording or (
                            self.debug
                            and current_frame_index % max(1, visualize_every) == 0
                        ):
                            global_prediction = local_to_global[prediction]

                    if self.compute_optical_flow and prev_frame is not None:
                        backend_val = str(self.optical_flow_backend).lower()
                        use_raft = "raft" in backend_val
                        use_torch_farneback = ("torch" in backend_val) and not use_raft
                        self._flow_hsv, self._flow = compute_optical_flow(
                            prev_frame,
                            frame,
                            use_raft=use_raft,
                            use_torch_farneback=use_torch_farneback,
                            **self._optical_flow_kwargs,
                        )

                    shape_notes_for_frame: Dict[str, str] = {}
                    # CUTIE can occasionally omit a seeded object on the first step.
                    # Backfill directly from the seed mask to keep expected instances
                    # stable and prime fallback history for subsequent frames.
                    if current_frame_index == segment.start_frame:
                        seed_mask_dict = self._build_seed_mask_dict(
                            segment, instance_names
                        )
                        missing_seed_labels = instance_names - set(mask_dict.keys())
                        if missing_seed_labels:
                            for missing_label in sorted(missing_seed_labels):
                                seed_mask = seed_mask_dict.get(missing_label)
                                if seed_mask is None:
                                    continue
                                mask_dict[missing_label] = seed_mask.copy()
                                shape_notes_for_frame[missing_label] = (
                                    "filled_from_seed_mask(start_frame)"
                                )

                    if len(mask_dict) < expected_instance_count:
                        missing_instances = instance_names - set(mask_dict.keys())
                        if missing_instances:
                            missing_count = int(expected_instance_count - len(mask_dict))
                            verb = "is" if missing_count == 1 else "are"
                            noun = "instance" if missing_count == 1 else "instances"
                            message = (
                                f"There {verb} {missing_count} missing {noun} in the current frame ({current_frame_index}).\n\n"
                                f"Missing or occluded: {', '.join(str(instance) for instance in missing_instances)}"
                            )
                            message_with_index = (
                                message + delimiter + str(current_frame_index)
                            )
                            missing_key = tuple(
                                sorted(str(instance) for instance in missing_instances)
                            )
                            if missing_key == last_missing_key:
                                missing_streak += 1
                            else:
                                last_missing_key = missing_key
                                missing_streak = 1
                                suppressed_missing_logs = 0

                            should_log_missing = (
                                missing_streak == 1
                                or missing_streak % missing_log_every == 0
                            )
                            if should_log_missing:
                                if suppressed_missing_logs > 0:
                                    logger.info(
                                        "Suppressed %s repetitive missing-instance log(s) for: %s",
                                        suppressed_missing_logs,
                                        ", ".join(missing_key),
                                    )
                                    suppressed_missing_logs = 0
                                logger.info(message)
                            else:
                                suppressed_missing_logs += 1

                            if self.auto_missing_instance_recovery:
                                recovered_instances, _ = (
                                    self._recover_missing_instances_with_bbox(
                                        missing_instances, frame
                                    )
                                )
                                if recovered_instances:
                                    mask_dict.update(recovered_instances)
                            missing_instances = instance_names - set(mask_dict.keys())
                            if missing_instances:
                                filled_instances, filled_notes = (
                                    self._fill_missing_instances_from_recent_masks(
                                        missing_instances
                                    )
                                )
                                if filled_instances:
                                    mask_dict.update(filled_instances)
                                    shape_notes_for_frame.update(filled_notes)
                                    missing_instances = (
                                        instance_names - set(mask_dict.keys())
                                    )

                            if missing_instances:
                                logger.info(
                                    "Missing instances after recovery/fill at frame %s: %s",
                                    current_frame_index,
                                    ", ".join(
                                        str(instance)
                                        for instance in sorted(missing_instances)
                                    ),
                                )

                            if (
                                missing_instances
                                and not has_occlusion
                                and not self.continue_on_missing_instances
                            ):
                                self._save_annotation_with_notes(
                                    filename,
                                    mask_dict,
                                    frame.shape,
                                    shape_notes=shape_notes_for_frame,
                                )
                                saved_mask_dict = getattr(
                                    self, "_last_saved_instance_masks", {}
                                )
                                if not isinstance(saved_mask_dict, dict):
                                    saved_mask_dict = {}
                                self._update_recent_instance_masks(
                                    current_frame_index, saved_mask_dict
                                )
                                if pred_worker is not None:
                                    pred_worker.stop_signal.emit()
                                return (message_with_index, True)

                    self._save_annotation_with_notes(
                        filename,
                        mask_dict,
                        frame.shape,
                        shape_notes=shape_notes_for_frame,
                    )
                    saved_mask_dict = getattr(self, "_last_saved_instance_masks", {})
                    if not isinstance(saved_mask_dict, dict):
                        saved_mask_dict = {}
                    self._update_recent_instance_masks(
                        current_frame_index, saved_mask_dict
                    )

                    if recording:
                        if global_prediction is None:
                            global_prediction = local_to_global[prediction]
                        self._mask = global_prediction > 0
                        visualization = overlay_davis(frame, global_prediction)
                        if self._flow_hsv is not None:
                            flow_bgr = cv2.cvtColor(self._flow_hsv, cv2.COLOR_HSV2BGR)
                            expanded_prediction = np.expand_dims(self._mask, axis=-1)
                            flow_bgr = flow_bgr * expanded_prediction
                            mask_array_reshaped = np.repeat(
                                self._mask[:, :, np.newaxis], 2, axis=2
                            )
                            visualization = cv2.addWeighted(
                                visualization, 1, flow_bgr, 0.5, 0
                            )
                            visualization = draw.draw_flow(
                                visualization, self._flow * mask_array_reshaped
                            )
                        self.video_writer.write(visualization)

                    if (
                        self.debug
                        and current_frame_index % max(1, visualize_every) == 0
                    ):
                        if global_prediction is None:
                            global_prediction = local_to_global[prediction]
                        self.save_color_id_mask(frame, global_prediction, filename)

                    if self.compute_optical_flow:
                        prev_frame = frame.copy()
                    current_frame_index += 1

        message = "Stop at frame:\n" + delimiter + str(current_frame_index - 1)
        return (message, False)

    def process_video_with_mask(
        self,
        frame_number=0,
        mask=None,
        frames_to_propagate=60,
        visualize_every=30,
        labels_dict=None,
        pred_worker=None,
        recording=False,
        output_video_path=None,
        has_occlusion=False,
        seed_frames: Optional[List[SeedFrame]] = None,
        end_frame_number: Optional[int] = None,
    ):
        if mask is None:
            return "No initial mask provided for CUTIE propagation."

        if labels_dict is None:
            labels_dict = {"_background_": 0}
        elif "_background_" not in labels_dict:
            labels_dict = {"_background_": 0, **labels_dict}

        self.label_registry = {"_background_": 0}
        self._committed_seed_frames.clear()
        self._seed_segment_lookup = {}
        self._video_active_object_ids = set()
        self._recent_instance_masks.clear()
        self._recent_instance_mask_frames.clear()
        self._seed_frames = seed_frames or []

        ordered_labels = sorted(labels_dict.items(), key=lambda item: item[1])
        original_map = {label: value for label, value in ordered_labels}
        if "_background_" not in original_map:
            original_map["_background_"] = 0

        global_label_map = self._assign_global_ids(original_map)
        remapped_mask = self._remap_mask_to_global(
            mask.astype(np.int32), original_map, global_label_map
        )
        active_labels = self._extract_active_labels(remapped_mask, global_label_map)

        segment_end = end_frame_number
        if segment_end is None and frames_to_propagate is not None:
            segment_end = frame_number + frames_to_propagate

        segment = SeedSegment(
            seed=None,
            start_frame=frame_number,
            end_frame=segment_end,
            mask=remapped_mask,
            labels_map=global_label_map,
            active_labels=active_labels,
        )

        return self._run_segments(
            segments=[segment],
            pred_worker=pred_worker,
            recording=recording,
            output_video_path=output_video_path,
            has_occlusion=has_occlusion,
            seed_frames=self._seed_frames,
            seed_segment_lookup=self._seed_segment_lookup,
            visualize_every=visualize_every,
        )

    def process_video_from_seeds(
        self,
        end_frame: Optional[int] = None,
        start_frame: int = 0,
        pred_worker=None,
        recording: bool = False,
        output_video_path: Optional[str] = None,
        has_occlusion: bool = False,
        visualize_every: int = 30,
    ) -> str:
        """Process the video using discovered seed frames."""

        self.label_registry = {"_background_": 0}
        self._committed_seed_frames.clear()
        self._video_active_object_ids = set()
        self._recent_instance_masks.clear()
        self._recent_instance_mask_frames.clear()
        self._seed_frames = self.discover_seed_frames(
            self.video_name, self.video_folder, force_refresh=True
        )
        if not self._seed_frames:
            message = (
                "No label frames found. Please label a frame click save  "
                "(PNG+JSON are saved together) before running CUTIE."
            )
            logger.info(message)
            self._emit_stop_signal(pred_worker)
            return message

        self._ensure_seed_cache(self._seed_frames)
        if not self._seed_segment_lookup:
            message = "No valid seed masks were parsed for this video."
            logger.info(message)
            self._emit_stop_signal(pred_worker)
            return message

        try:
            normalized_start_frame = max(0, int(start_frame))
        except (TypeError, ValueError):
            normalized_start_frame = 0
        self._seed_frames = self._select_seed_frames_for_start(
            start_frame=normalized_start_frame
        )
        if not self._seed_frames:
            message = "No seed frames available at or after the requested start frame."
            logger.info(message)
            self._emit_stop_signal(pred_worker)
            return message

        self.num_tracking_instances = self._count_tracking_instances(
            self._seed_frames
        )

        segments = self._build_seed_segments(self._seed_frames, end_frame)
        if not segments:
            message = "No valid seed segments were generated for CUTIE processing."
            logger.info(message)
            self._emit_stop_signal(pred_worker)
            return message

        return self._run_segments(
            segments=segments,
            pred_worker=pred_worker,
            recording=recording,
            output_video_path=output_video_path,
            has_occlusion=has_occlusion,
            seed_frames=self._seed_frames,
            seed_segment_lookup=self._seed_segment_lookup,
            visualize_every=visualize_every,
        )

    @staticmethod
    def _emit_stop_signal(pred_worker) -> None:
        if pred_worker is None:
            return
        try:
            pred_worker.stop_signal.emit()
        except Exception:
            pass

    def _select_seed_frames_for_start(self, start_frame: int) -> List[SeedFrame]:
        valid_seed_frames = [
            seed
            for seed in self._seed_frames
            if seed.frame_index in self._seed_segment_lookup
        ]
        valid_seed_frames = sorted(valid_seed_frames, key=lambda seed: seed.frame_index)

        if not valid_seed_frames:
            return []

        prior_seeds = [
            seed for seed in valid_seed_frames if seed.frame_index < start_frame
        ]
        later_seeds = [
            seed for seed in valid_seed_frames if seed.frame_index >= start_frame
        ]

        # Include the nearest earlier seed so CUTIE can propagate through the gap
        # before the first later seed (e.g., seeds at 0 and 100 with start at 1).
        if prior_seeds:
            nearest_prior_seed = prior_seeds[-1]
            selected = [nearest_prior_seed, *later_seeds]
            # Deduplicate while preserving order.
            deduped = []
            seen = set()
            for seed in selected:
                if seed.frame_index in seen:
                    continue
                seen.add(seed.frame_index)
                deduped.append(seed)
            logger.info(
                "Using nearest earlier seed at frame %s for start_frame %s.",
                nearest_prior_seed.frame_index,
                start_frame,
            )
            return deduped

        if later_seeds:
            return later_seeds

        return []

    def _count_tracking_instances(self, seeds: List[SeedFrame]) -> int:
        object_ids: Set[int] = set()
        for seed in seeds:
            segment = self._seed_segment_lookup.get(seed.frame_index)
            if segment is None:
                continue
            for label in segment.active_labels:
                value = segment.labels_map.get(label)
                if value is None or int(value) == 0:
                    continue
                object_ids.add(int(value))
        return len(object_ids)

    def save_color_id_mask(self, frame, prediction, filename):
        _id_mask = color_id_mask(prediction)
        visualization = overlay_davis(frame, prediction)
        # Convert RGB to BGR
        visualization_rgb = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        # Show the image
        cv2.imwrite(str(filename).replace(".json", ".png"), frame)
        cv2.imwrite(
            str(filename).replace(".json", "_mask_frame.png"), visualization_rgb
        )
        cv2.imwrite(str(filename).replace(".json", "_mask.png"), _id_mask)


if __name__ == "__main__":
    # Example usage:
    video_name = "demo/video.mp4"
    mask_name = "demo/first_frame.png"
    video_folder = video_name.split(".")[0]
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    mask = np.array(Image.open(mask_name))
    labels_dict = {
        "object_1": 1,
        "object_2": 2,
        "object_3": 3,
    }  # Example labels dictionary
    processor = CutieCoreVideoProcessor(video_name, debug=True)
    processor.process_video_with_mask(
        mask=mask, visualize_every=30, frames_to_propagate=30, labels_dict=labels_dict
    )
