import os
import cv2
import torch
import gdown
import json
import numpy as np
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set
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
from shapely.geometry import Polygon
from omegaconf import open_dict
from hydra import compose, initialize
from annolid.segmentation.cutie_vos.model.cutie import CUTIE
from annolid.segmentation.cutie_vos.inference.inference_core import InferenceCore
from labelme.utils.shape import shapes_to_label
from annolid.utils.shapes import extract_flow_points_in_mask
from annolid.utils.devices import get_device
from annolid.utils.logger import logger
from annolid.motion.optical_flow import compute_optical_flow
from annolid.utils import draw
from annolid.utils.files import create_tracking_csv_file
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


class CutieVideoProcessor:

    _REMOTE_MODEL_URL = "https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth"
    _MD5 = "a6071de6136982e396851903ab4c083a"

    def __init__(self, video_name, *args, **kwargs):
        self.video_name = video_name
        results_folder = kwargs.get('results_folder')
        self.video_folder = Path(results_folder) if results_folder else Path(
            video_name).with_suffix("")
        self.results_folder = self.video_folder
        self.mem_every = kwargs.get('mem_every', 5)
        self.debug = kwargs.get('debug', False)
        # T_max parameter default 5
        self.max_mem_frames = kwargs.get('t_max_value', 5)
        self.use_cpu_only = kwargs.get('use_cpu_only', False)
        self.epsilon_for_polygon = kwargs.get('epsilon_for_polygon', 2.0)
        self.processor = None
        self.num_tracking_instances = 0
        current_file_path = os.path.abspath(__file__)
        self.current_folder = os.path.dirname(current_file_path)
        self.device = 'cpu' if self.use_cpu_only else get_device()
        logger.info(f"Running device: {self.device}.")
        self.cutie, self.cfg = self._initialize_model()
        logger.info(
            f"Using epsilon: {self.epsilon_for_polygon} for polygon approximation.")

        self._frame_numbers = []
        self._instance_names = []
        self._cx_values = []
        self._cy_values = []
        self._motion_indices = []
        self.output_tracking_csvpath = None
        self._frame_number = None
        self._motion_index = ''
        self._instance_name = ''
        self._flow = None
        self._flow_hsv = None
        self._mask = None
        self.cache = BboxCache(max_size=self.mem_every * 10)
        self.sam_hq = None
        self.output_tracking_csvpath = str(
            self.video_folder) + f"_tracked.csv"
        self.showing_KMedoids_in_mask = False
        self.compute_optical_flow = kwargs.get('compute_optical_flow', False)
        self.optical_flow_backend = kwargs.get(
            'optical_flow_backend', 'farneback')
        self.auto_missing_instance_recovery = kwargs.get(
            "auto_missing_instance_recovery", False)
        logger.info(
            f"Auto missing instance recovery is set to {self.auto_missing_instance_recovery}.")
        self._seed_frames: List[SeedFrame] = []
        self.label_registry: Dict[str, int] = {"_background_": 0}
        self._seed_segment_lookup: Dict[int, SeedSegment] = {}
        self._committed_seed_frames: Set[int] = set()
        self._cached_labeled_frames: Optional[Set[int]] = None
        self._seed_object_counts: Dict[int, int] = {}
        self._global_object_ids: List[int] = []
        self._global_label_names: Dict[int, str] = {}
        self._video_seed_cache: Dict[str, Dict[str, Any]] = {}
        self._current_video_cache_key: Optional[str] = None
        self._video_active_object_ids: Set[int] = set()

    def set_same_hq(self, sam_hq):
        self.sam_hq = sam_hq

    @staticmethod
    def discover_seed_frames(video_name, results_folder: Optional[Path] = None) -> List[SeedFrame]:
        """Discover seed frame pairs (PNG+JSON) within the video results folder."""
        results_dir = Path(results_folder) if results_folder else Path(
            video_name).with_suffix("")
        if not results_dir.exists() or not results_dir.is_dir():
            logger.info(
                f"CUTIE seed discovery skipped: folder does not exist -> {results_dir}")
            return []

        stem = results_dir.name
        stem_prefix = f"{stem}_"
        stem_prefix_lower = stem_prefix.lower()

        def collect_seeds(directory: Path) -> Dict[int, SeedFrame]:
            collected: Dict[int, SeedFrame] = {}
            if not directory.exists() or not directory.is_dir():
                logger.debug(
                    f"Seed scan skipped: not a directory -> {directory}")
                return collected

            png_candidates = sorted(directory.glob('*.png'))
            logger.info(
                f"Seed scan in {directory} found {len(png_candidates)} png candidate(s)")

            for png_path in png_candidates:
                name_lower = png_path.stem.lower()
                if not name_lower.startswith(stem_prefix_lower):
                    logger.debug(
                        f"Skipping {png_path.name}: stem prefix mismatch")
                    continue

                suffix = name_lower[len(stem_prefix_lower):]
                if len(suffix) != 9 or not suffix.isdigit():
                    logger.debug(
                        f"Skipping {png_path.name}: expected 9-digit suffix, got '{suffix}'")
                    continue

                frame_index = int(suffix)

                json_path = png_path.with_suffix('.json')
                if not json_path.exists():
                    logger.debug(
                        f"Skipping {png_path.name}: missing JSON {json_path.name}")
                    continue

                existing = collected.get(frame_index)
                if existing and existing.png_path.stat().st_mtime >= png_path.stat().st_mtime:
                    logger.debug(
                        f"Skipping {png_path.name}: older than registered seed for frame {frame_index}")
                    continue

                collected[frame_index] = SeedFrame(
                    frame_index=frame_index,
                    png_path=png_path,
                    json_path=json_path,
                )
                logger.info(
                    f"Registered CUTIE seed {png_path.name} (frame {frame_index})")

            return collected

        logger.info(f"Scanning for CUTIE seeds in {results_dir}")
        seeds = collect_seeds(results_dir)

        nested_dir = results_dir / stem
        if not seeds and nested_dir.exists():
            logger.info(
                f"No seeds found at root; scanning nested directory {nested_dir}")
            seeds = collect_seeds(nested_dir)

        discovered = [seeds[idx] for idx in sorted(seeds.keys())]
        logger.info(
            f"Discovered {len(discovered)} CUTIE seed(s) in {results_dir}")
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
                    (shape.get('shape_type') == 'polygon' and len(
                        shape.get('points', [])) >= 3)
                    or shape.get('mask')
                ):
                    return True
            return False

        # Prefer the annotation store (written during prediction) because per-frame
        # JSON files are not persisted for auto-labeled frames.
        store_path = results_dir / \
            f"{results_dir.name}{AnnotationStore.STORE_SUFFIX}"
        if store_path.exists():
            try:
                store = AnnotationStore(store_path)
                records = store._load_records()
                for frame_idx, record in records.items():
                    try:
                        normalized_idx = int(frame_idx)
                    except (TypeError, ValueError):
                        continue
                    if has_valid_shapes(record.get('shapes')):
                        labeled_frames.add(normalized_idx)
            except Exception as exc:
                logger.debug(
                    "Failed to read annotation store %s: %s", store_path, exc
                )

        def scan_directory(directory: Path):
            if not directory.exists() or not directory.is_dir():
                return
            for json_path in directory.glob('*.json'):
                name_lower = json_path.stem.lower()
                if not name_lower.startswith(stem_prefix_lower):
                    continue
                suffix = name_lower[len(stem_prefix_lower):]
                if len(suffix) != 9 or not suffix.isdigit():
                    continue
                frame_idx = int(suffix)
                # Skip JSON files for frames already accounted for via the store.
                if frame_idx in labeled_frames:
                    continue
                try:
                    with open(json_path, 'r', encoding='utf-8') as fp:
                        data = json.load(fp) or {}
                except (OSError, ValueError) as exc:
                    logger.debug(
                        f"Failed to parse JSON annotation {json_path.name}: {exc}")
                    continue

                shapes = data.get('shapes') or []
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

            for label, value in segment.labels_map.items():
                if label == "_background_":
                    continue
                global_label_names.setdefault(value, label)
                global_object_ids.add(value)

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
        cache_frames = cache_entry.get(
            "frame_indices") if cache_entry else None

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
    def _segment_already_completed(segment: SeedSegment,
                                   resolved_end: int,
                                   labeled_frames: Set[int]) -> bool:
        """Return True if every frame in the segment range already has annotations."""
        if resolved_end is None:
            return False
        if resolved_end < segment.start_frame:
            return True

        for frame_idx in range(segment.start_frame, resolved_end + 1):
            if frame_idx not in labeled_frames:
                return False
        return True

    def initialize_video_writer(self, output_video_path,
                                frame_width,
                                frame_height,
                                fps=30
                                ):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_video_path, fourcc, fps, (frame_width, frame_height))

    def _initialize_model(self):
        # general setup
        torch.cuda.empty_cache()
        with torch.inference_mode():
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            initialize(version_base='1.3.2', config_path="config",
                       job_name="eval_config")
            cfg = compose(config_name="eval_config")
            model_path = os.path.join(
                self.current_folder, 'weights/cutie-base-mega.pth')
            if not os.path.exists(model_path):
                gdown.cached_download(self._REMOTE_MODEL_URL,
                                      model_path,
                                      md5=self._MD5
                                      )
            with open_dict(cfg):
                cfg['weights'] = model_path
                cfg['max_mem_frames'] = self.max_mem_frames
            cfg['mem_every'] = self.mem_every
            logger.info(
                f"Saving into working memeory for every: {self.mem_every}.")
            logger.info(f"Tmax: max_mem_frames: {self.max_mem_frames}")
            cutie_model = CUTIE(cfg).to(self.device).eval()
            model_weights = torch.load(
                cfg.weights, map_location=self.device, weights_only=True)
            cutie_model.load_weights(model_weights)
        return cutie_model, cfg

    def _save_bbox(self, points, frame_area, label):
        # A linearring requires at least 4 coordinates.
        # good quality polygon
        if len(points) >= 4:
            # Create a Shapely Polygon object from the list of points
            polygon = Polygon(points)
            # Get the bounding box coordinates (minx, miny, maxx, maxy)
            _bbox = polygon.bounds
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
            return
        self._instance_names.append(label)
        self._frame_numbers.append(self._frame_number)
        self._cx_values.append(cx)
        self._cy_values.append(cy)
        if self._flow_hsv is not None:
            # unnormalized magnitude
            magnitude = self._flow_hsv[..., 2]
            magnitude = magnitude.astype(np.float32)
            mask_sum = np.sum(mask)
            if mask_sum > 0:
                self._motion_index = np.sum(
                    mask * magnitude) / mask_sum
            else:
                self._motion_index = 0.0
        else:
            self._motion_index = -1
        self._motion_indices.append(self._motion_index)

    def _save_annotation(self, filename, mask_dict, frame_shape):
        height, width, _ = frame_shape
        frame_area = height * width
        label_list = []
        for label_id, mask in mask_dict.items():
            label = str(label_id)

            self._save_results(label, mask)
            self.save_KMedoids_in_mask(label_list, mask)

            current_shape = MaskShape(label=label,
                                      flags={},
                                      description=f'motion_index: {self._motion_index}',)
            current_shape.mask = mask
            _shapes = current_shape.toPolygons(
                epsilon=self.epsilon_for_polygon)
            if len(_shapes) < 0:
                continue
            current_shape = _shapes[0]
            points = [[point.x(), point.y()] for point in current_shape.points]
            self._save_bbox(points, frame_area, label)
            current_shape.points = points
            label_list.append(current_shape)
        save_labels(filename=filename, imagePath=None, label_list=label_list,
                    height=height, width=width, save_image_to_json=False,
                    persist_json=False)
        return label_list

    def save_KMedoids_in_mask(self, label_list, mask):
        if self._flow is not None and self.showing_KMedoids_in_mask:
            flow_points = extract_flow_points_in_mask(mask, self._flow)
            for fpoint in flow_points.tolist():
                fpoint_shape = Shape(label='kmedoids',
                                     flags={},
                                     shape_type='point',
                                     description="kmedoids of flow in mask"
                                     )
                fpoint_shape.points = [fpoint]
                label_list.append(fpoint_shape)

    def segement_with_bbox(self, instance_names, cur_frame, score_threshold=0.60):
        label_mask_dict = {}
        for instance_name in instance_names:
            _bboxes = self.cache.get_most_recent_bbox(instance_name)
            if _bboxes is not None:
                masks, scores, input_box = self.sam_hq.segment_objects(
                    cur_frame, [_bboxes])
                logger.info(
                    f"Use bbox prompt to recover {instance_name} with score {scores}.")
                logger.info(f"Using score threshold: {score_threshold} ")
                if scores[0] > score_threshold:
                    label_mask_dict[instance_name] = masks[0]
        return label_mask_dict

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
        with open(label_json_file, 'r') as json_file:
            data = json.load(json_file)

        filtered_shapes = []
        for shape in data.get('shapes', []):
            points = shape.get('points') or []
            if len(points) < 3:
                continue

            label_text = (shape.get('label') or '').lower()
            description_text = (shape.get('description') or '').lower()
            if 'zone' in label_text or 'zone' in description_text:
                continue

            flags = shape.get('flags') or {}
            if any(str(flag_key).lower() == 'zone' and bool(flag_val)
                   for flag_key, flag_val in flags.items()):
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

        mask, _ = shapes_to_label(
            image_size, filtered_shapes, label_name_to_value)
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
    def _remap_mask_to_global(mask: np.ndarray,
                              original_label_map: Dict[str, int],
                              global_label_map: Dict[str, int]) -> np.ndarray:
        remapped = np.zeros_like(mask, dtype=np.int32)
        for label, original_value in original_label_map.items():
            global_value = global_label_map.get(label, 0)
            remapped[mask == original_value] = global_value
        return remapped

    @staticmethod
    def _extract_active_labels(mask: np.ndarray,
                               global_label_map: Dict[str, int]) -> List[str]:
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
        one_hot = index_numpy_to_one_hot_torch(
            mask.astype(np.int64), num_classes)
        selected = one_hot[unique_ids]
        return selected, unique_ids

    @staticmethod
    def _map_local_prediction_to_global(prediction: np.ndarray,
                                        active_global_ids: List[int]) -> np.ndarray:
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

        mask, label_map = self.shapes_to_mask(
            str(seed.json_path), frame.shape[:2])
        if mask is None or len(np.unique(mask)) <= 1:
            logger.warning(
                f"Seed {seed.png_path.name} has no valid polygon annotations; skipping.")
            return None

        if "_background_" not in label_map:
            label_map = {"_background_": 0, **label_map}
        global_label_map = self._assign_global_ids(label_map)
        remapped_mask = self._remap_mask_to_global(
            mask, label_map, global_label_map)
        active_labels = self._extract_active_labels(
            remapped_mask, global_label_map)

        return SeedSegment(
            seed=seed,
            start_frame=seed.frame_index,
            end_frame=None,
            mask=remapped_mask,
            labels_map=global_label_map,
            active_labels=active_labels,
        )

    def _build_seed_segments(self,
                             seeds: List[SeedFrame],
                             requested_end: Optional[int]) -> List[SeedSegment]:
        """Create contiguous segments from discovered seeds."""
        segments: List[SeedSegment] = []
        for idx, seed in enumerate(seeds):
            cached_segment = self._seed_segment_lookup.get(seed.frame_index)
            if cached_segment is None:
                cached_segment = self._load_seed_mask(seed)
                if cached_segment is None:
                    continue
                self._seed_segment_lookup[seed.frame_index] = cached_segment
                self._seed_object_counts[seed.frame_index] = len(
                    cached_segment.active_labels)
                for label, value in cached_segment.labels_map.items():
                    if label == "_background_":
                        continue
                    if value not in self._global_label_names:
                        self._global_label_names[value] = label
                        self._global_object_ids.append(value)
                self._global_object_ids.sort()

            segment = replace(cached_segment)
            segment.start_frame = seed.frame_index
            segment.end_frame = None

            next_seed_frame = None
            if idx + 1 < len(seeds):
                next_seed_frame = seeds[idx + 1].frame_index

            if requested_end is not None:
                segment.end_frame = requested_end
            if next_seed_frame is not None:
                candidate_end = next_seed_frame - 1
                if segment.end_frame is None:
                    segment.end_frame = candidate_end
                else:
                    segment.end_frame = min(segment.end_frame, candidate_end)

            if segment.end_frame is not None and segment.end_frame < segment.start_frame:
                logger.info(
                    f"Skipping degenerate seed range [{segment.start_frame}, {segment.end_frame}].")
                continue

            segments.append(segment)

        if requested_end is not None and segments:
            # Ensure the final segment honours requested_end.
            segments[-1].end_frame = min(
                requested_end, segments[-1].end_frame
                if segments[-1].end_frame is not None else requested_end)

        return segments

    def commit_masks_into_permanent_memory(self, frame_number, labels_dict,
                                           seed_frames: Optional[List[SeedFrame]] = None,
                                           seed_segment_lookup: Optional[Dict[int,
                                                                              SeedSegment]] = None,
                                           segment_end: Optional[int] = None):
        """
        Commit masks into permanent memory for inference.

        Args:
            frame_number (int): Frame number.
            labels_dict (dict): Dictionary mapping label names to their values.

        Returns:
            dict: Updated labels dictionary.
        """
        with torch.inference_mode():
            with torch.amp.autocast('cuda', enabled=self.cfg.amp and self.device == 'cuda'):
                candidate_seeds = seed_frames or self._seed_frames
                if not candidate_seeds:
                    candidate_seeds = self.discover_seed_frames(
                        self.video_name, self.video_folder)

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
                        segment = self._seed_segment_lookup.get(
                            seed.frame_index)
                    if segment is None:
                        segment = self._load_seed_mask(seed)
                        if segment and segment.seed is not None:
                            self._seed_segment_lookup[segment.seed.frame_index] = segment
                    if segment is None:
                        continue

                    if not segment.active_labels:
                        continue

                    frame = cv2.imread(
                        str(segment.seed.png_path)) if segment.seed else None
                    if frame is None:
                        logger.warning(
                            f"Failed to read seed frame for committing permanent memory: {segment.seed}")
                        continue

                    for label in segment.active_labels:
                        labels_dict.setdefault(
                            label, segment.labels_map[label])

                    frame_torch = image_to_torch(frame, device=self.device)
                    mask_tensor, active_ids = self._build_object_mask_tensor(
                        segment.mask)
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
                        f"Committed {len(active_ids)} instances from seed #{segment.seed.frame_index} into permanent memory.")

                return labels_dict

    def _run_segments(self,
                      segments: List[SeedSegment],
                      pred_worker=None,
                      recording: bool = False,
                      output_video_path: Optional[str] = None,
                      has_occlusion: bool = False,
                      seed_frames: Optional[List[SeedFrame]] = None,
                      seed_segment_lookup: Optional[Dict[int,
                                                         SeedSegment]] = None,
                      visualize_every: int = 30) -> str:
        if not segments:
            return "No valid segments found for CUTIE processing."

        # Refresh cached labeled frames so resume logic sees the latest edits
        self._cached_labeled_frames = None
        labeled_frames = self._collect_labeled_frame_indices()

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
                output_video_path = self.output_tracking_csvpath.replace(
                    '.csv', '.mp4')
            self.initialize_video_writer(
                output_video_path, frame_width, frame_height, fps)
            logger.info(f"Saving the color masks to video {output_video_path}")

        final_message: Optional[str] = None
        halt_requested = False
        processed_any_segment = False
        skipped_segments = 0

        if seed_segment_lookup is None:
            seed_segment_lookup = {
                segment.seed.frame_index: segment
                for segment in segments if segment.seed is not None
            }

        try:
            for idx, segment in enumerate(segments):
                if pred_worker is not None and pred_worker.is_stopped():
                    halt_requested = True
                    break

                resolved_end = segment.end_frame
                if resolved_end is None:
                    resolved_end = last_frame_index
                else:
                    resolved_end = min(resolved_end, last_frame_index)

                if resolved_end < segment.start_frame:
                    logger.info(
                        f"Skipping segment starting at {segment.start_frame} with end {resolved_end}.")
                    continue

                if self._segment_already_completed(segment, resolved_end, labeled_frames):
                    skipped_segments += 1
                    logger.info(
                        f"Skipping Cutie segment [{segment.start_frame}, {resolved_end}] - annotations already exist.")
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

            if not processed_any_segment and skipped_segments > 0 and final_message is None:
                final_message = ("CUTIE processing skipped. All selected segment ranges already "
                                 "contained saved annotations.")

            if final_message is None:
                last_segment = segments[-1]
                resolved_end = last_segment.end_frame
                if resolved_end is None:
                    resolved_end = last_frame_index
                final_message = ("Stop at frame:\n" +
                                 f"#{max(last_segment.start_frame, resolved_end)}")
        finally:
            cap.release()
            if recording and hasattr(self, 'video_writer'):
                self.video_writer.release()

        try:
            if fps > 0:
                create_tracking_csv_file(self._frame_numbers,
                                         self._instance_names,
                                         self._cx_values,
                                         self._cy_values,
                                         self._motion_indices,
                                         self.output_tracking_csvpath,
                                         self.video_name,
                                         fps)
        except Exception as exc:
            logger.error(f"Failed to save tracking CSV: {exc}")

        if pred_worker is not None and not halt_requested:
            pred_worker.stop_signal.emit()

        return final_message or "CUTIE processing completed."

    def _process_segment(self,
                         cap,
                         segment: SeedSegment,
                         end_frame: int,
                         fps: float,
                         pred_worker=None,
                         recording: bool = False,
                         has_occlusion: bool = False,
                         seed_frames: Optional[List[SeedFrame]] = None,
                         seed_segment_lookup: Optional[Dict[int,
                                                            SeedSegment]] = None,
                         visualize_every: int = 30) -> (Optional[str], bool):
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
            return (f"No valid polygon found in seed frame #{segment.start_frame}", True)

        mask_tensor = mask_tensor.to(self.device)
        self._register_active_objects(active_ids)
        value_to_label_names = {
            value: label for label, value in self.label_registry.items()
        }
        if self._global_label_names:
            value_to_label_names.update(self._global_label_names)
        instance_names = set(segment.active_labels)
        local_to_global = np.zeros(len(active_ids) + 1, dtype=np.int32)
        for local_idx, global_id in enumerate(active_ids, start=1):
            local_to_global[local_idx] = int(global_id)

        current_frame_index = segment.start_frame
        prev_frame = None
        delimiter = '#'

        if current_frame_index >= end_frame:
            # Still process the single frame
            end_frame = current_frame_index

        with torch.inference_mode():
            with torch.amp.autocast('cuda', enabled=self.cfg.amp and self.device == 'cuda'):
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != current_frame_index:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
                while cap.isOpened():
                    if pred_worker is not None and pred_worker.is_stopped():
                        return (None, True)

                    if current_frame_index > end_frame:
                        break

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    self._frame_number = current_frame_index
                    frame_torch = image_to_torch(frame, device=self.device)
                    filename = self.video_folder / (
                        self.video_folder.name +
                        f"_{current_frame_index:0>{9}}.json")

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

                    if prediction is None:
                        global_prediction = mask.copy()
                        mask_dict = {}
                        for global_id in active_ids:
                            global_mask = global_prediction == int(global_id)
                            if not global_mask.any():
                                continue
                            label_name = value_to_label_names.get(
                                int(global_id), str(global_id))
                            mask_dict[label_name] = global_mask
                    else:
                        mask_dict = {}
                        for local_idx, global_id in enumerate(active_ids, start=1):
                            local_mask = prediction == local_idx
                            if not local_mask.any():
                                continue
                            label_name = value_to_label_names.get(
                                int(global_id), str(global_id))
                            mask_dict[label_name] = local_mask

                        global_prediction = None
                        if (
                            recording
                            or (
                                self.debug
                                and current_frame_index % max(1, visualize_every) == 0
                            )
                        ):
                            global_prediction = local_to_global[prediction]

                    if self.compute_optical_flow and prev_frame is not None:
                        backend_val = str(self.optical_flow_backend).lower()
                        use_raft = "raft" in backend_val
                        use_torch_farneback = (
                            "torch" in backend_val) and not use_raft
                        self._flow_hsv, self._flow = compute_optical_flow(
                            prev_frame,
                            frame,
                            use_raft=use_raft,
                            use_torch_farneback=use_torch_farneback,
                        )

                    self._save_annotation(filename, mask_dict, frame.shape)

                    if len(mask_dict) < self.num_tracking_instances:
                        missing_instances = instance_names - \
                            set(mask_dict.keys())
                        if missing_instances:
                            message = (
                                f"There are {self.num_tracking_instances - len(mask_dict)} missing instance(s) in the current frame ({current_frame_index}).\n\n"
                                f"Missing or occluded: {', '.join(str(instance) for instance in missing_instances)}"
                            )
                            message_with_index = message + \
                                delimiter + str(current_frame_index)
                            logger.info(message)

                            if self.auto_missing_instance_recovery:
                                segemented_instances = self.segement_with_bbox(
                                    missing_instances, frame)
                                if len(segemented_instances) >= 1:
                                    mask_dict.update(segemented_instances)
                                    self._save_annotation(
                                        filename, mask_dict, frame.shape)

                            if len(mask_dict) < self.num_tracking_instances and not has_occlusion:
                                if pred_worker is not None:
                                    pred_worker.stop_signal.emit()
                                return (message_with_index, True)

                    if recording:
                        if global_prediction is None:
                            global_prediction = local_to_global[prediction]
                        self._mask = global_prediction > 0
                        visualization = overlay_davis(frame, global_prediction)
                        if self._flow_hsv is not None:
                            flow_bgr = cv2.cvtColor(
                                self._flow_hsv, cv2.COLOR_HSV2BGR)
                            expanded_prediction = np.expand_dims(
                                self._mask, axis=-1)
                            flow_bgr = flow_bgr * expanded_prediction
                            mask_array_reshaped = np.repeat(
                                self._mask[:, :, np.newaxis], 2, axis=2)
                            visualization = cv2.addWeighted(
                                visualization, 1, flow_bgr, 0.5, 0)
                            visualization = draw.draw_flow(
                                visualization, self._flow * mask_array_reshaped)
                        self.video_writer.write(visualization)

                    if self.debug and current_frame_index % max(1, visualize_every) == 0:
                        if global_prediction is None:
                            global_prediction = local_to_global[prediction]
                        self.save_color_id_mask(
                            frame, global_prediction, filename)

                    if self.compute_optical_flow:
                        prev_frame = frame.copy()
                    current_frame_index += 1

        message = ("Stop at frame:\n" +
                   delimiter + str(current_frame_index - 1))
        return (message, False)

    def process_video_with_mask(self, frame_number=0,
                                mask=None,
                                frames_to_propagate=60,
                                visualize_every=30,
                                labels_dict=None,
                                pred_worker=None,
                                recording=False,
                                output_video_path=None,
                                has_occlusion=False,
                                seed_frames: Optional[List[SeedFrame]] = None,
                                end_frame_number: Optional[int] = None):
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
        self._seed_frames = seed_frames or []

        ordered_labels = sorted(
            labels_dict.items(), key=lambda item: item[1])
        original_map = {label: value for label, value in ordered_labels}
        if "_background_" not in original_map:
            original_map["_background_"] = 0

        global_label_map = self._assign_global_ids(original_map)
        remapped_mask = self._remap_mask_to_global(
            mask.astype(np.int32), original_map, global_label_map)
        active_labels = self._extract_active_labels(
            remapped_mask, global_label_map)

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

    def process_video_from_seeds(self,
                                 end_frame: Optional[int] = None,
                                 start_frame: int = 0,
                                 pred_worker=None,
                                 recording: bool = False,
                                 output_video_path: Optional[str] = None,
                                 has_occlusion: bool = False,
                                 visualize_every: int = 30) -> str:
        """Process the video using discovered seed frames."""

        self.label_registry = {"_background_": 0}
        self._committed_seed_frames.clear()
        self._video_active_object_ids = set()
        self._seed_frames = self.discover_seed_frames(
            self.video_name, self.video_folder)
        if not self._seed_frames:
            message = ("No label frames found. Please label a frame click save  "
                       "(PNG+JSON are saved together) before running CUTIE.")
            logger.info(message)
            if pred_worker is not None:
                pred_worker.stop_signal.emit()
            return message

        self._ensure_seed_cache(self._seed_frames)
        if not self._seed_segment_lookup:
            message = "No valid seed masks were parsed for this video."
            logger.info(message)
            if pred_worker is not None:
                pred_worker.stop_signal.emit()
            return message

        self._seed_frames = [
            seed for seed in self._seed_frames
            if seed.frame_index in self._seed_segment_lookup
            and seed.frame_index >= start_frame
        ]
        if not self._seed_frames:
            message = (
                "No seed frames available at or after the requested start frame."
            )
            logger.info(message)
            if pred_worker is not None:
                pred_worker.stop_signal.emit()
            return message

        self.num_tracking_instances = len(self._video_active_object_ids)

        segments = self._build_seed_segments(self._seed_frames, end_frame)
        if not segments:
            message = "No valid seed segments were generated for CUTIE processing."
            logger.info(message)
            if pred_worker is not None:
                pred_worker.stop_signal.emit()
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

    def save_color_id_mask(self, frame, prediction, filename):
        _id_mask = color_id_mask(prediction)
        visualization = overlay_davis(frame, prediction)
        # Convert RGB to BGR
        visualization_rgb = cv2.cvtColor(
            visualization, cv2.COLOR_RGB2BGR)
        # Show the image
        cv2.imwrite(str(filename).replace(
            '.json', '.png'), frame)
        cv2.imwrite(str(filename).replace(
            '.json', '_mask_frame.png'), visualization_rgb)
        cv2.imwrite(str(filename).replace(
            '.json', '_mask.png'), _id_mask)


if __name__ == '__main__':
    # Example usage:
    video_name = 'demo/video.mp4'
    mask_name = 'demo/first_frame.png'
    video_folder = video_name.split('.')[0]
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    mask = np.array(Image.open(mask_name))
    labels_dict = {'object_1': 1, 'object_2': 2,
                   'object_3': 3}  # Example labels dictionary
    processor = CutieVideoProcessor(video_name, debug=True)
    processor.process_video_with_mask(
        mask=mask, visualize_every=30, frames_to_propagate=30, labels_dict=labels_dict)
