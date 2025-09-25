import os
import cv2
import torch
import gdown
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
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
        self.auto_missing_instance_recovery = kwargs.get(
            "auto_missing_instance_recovery", False)
        logger.info(
            f"Auto missing instance recovery is set to {self.auto_missing_instance_recovery}.")
        self._seed_frames: List[SeedFrame] = []
        self.label_registry: Dict[str, int] = {"_background_": 0}
        self._seed_segment_lookup: Dict[int, SeedSegment] = {}
        self._committed_seed_frames: Set[int] = set()

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
                    height=height, width=width, save_image_to_json=False)
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
        self._seed_segment_lookup = {}
        for idx, seed in enumerate(seeds):
            segment = self._load_seed_mask(seed)
            if segment is None:
                continue

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
            if segment.seed is not None:
                self._seed_segment_lookup[segment.seed.frame_index] = segment

        if requested_end is not None and segments:
            # Ensure the final segment honours requested_end.
            segments[-1].end_frame = min(
                requested_end, segments[-1].end_frame
                if segments[-1].end_frame is not None else requested_end)

        return segments

    def commit_masks_into_permanent_memory(self, frame_number, labels_dict,
                                           seed_frames: Optional[List[SeedFrame]] = None,
                                           seed_segment_lookup: Optional[Dict[int, SeedSegment]] = None):
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

                    segment = None
                    if seed_segment_lookup:
                        segment = seed_segment_lookup.get(seed.frame_index)
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
                    mask_tensor = mask_tensor.to(self.device)
                    self.processor.step(
                        frame_torch, mask_tensor,
                        idx_mask=False,
                        force_permanent=True
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

                if message:
                    final_message = message

                if should_halt:
                    halt_requested = True
                    break

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
                seed_segment_lookup=seed_segment_lookup)
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
        self.num_tracking_instances = len(segment.active_labels)
        value_to_label_names = {
            value: label for label, value in self.label_registry.items()
        }
        instance_names = set(segment.active_labels)

        current_frame_index = segment.start_frame
        prev_frame = None
        delimiter = '#'

        if current_frame_index >= end_frame:
            # Still process the single frame
            end_frame = current_frame_index

        with torch.inference_mode():
            with torch.amp.autocast('cuda', enabled=self.cfg.amp and self.device == 'cuda'):
                while cap.isOpened():
                    if pred_worker is not None and pred_worker.is_stopped():
                        return (None, True)

                    if current_frame_index > end_frame:
                        break

                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
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
                                frame_torch, mask_tensor,
                                idx_mask=False,
                                force_permanent=True
                            )
                            prediction = torch_prob_to_numpy_mask(prediction)
                            global_prediction = self._map_local_prediction_to_global(
                                prediction, active_ids)
                        except Exception as exc:  # pragma: no cover - logging only
                            logger.info(exc)
                            global_prediction = mask.copy()
                    else:
                        prediction = self.processor.step(frame_torch)
                        prediction = torch_prob_to_numpy_mask(prediction)
                        global_prediction = self._map_local_prediction_to_global(
                            prediction, active_ids)

                    if self.compute_optical_flow and prev_frame is not None:
                        self._flow_hsv, self._flow = compute_optical_flow(
                            prev_frame, frame)

                    mask_dict = {}
                    for label_id in np.unique(global_prediction):
                        if label_id == 0:
                            continue
                        label_name = value_to_label_names.get(
                            int(label_id), str(label_id))
                        mask_dict[label_name] = (global_prediction == label_id)

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
                        self.save_color_id_mask(
                            frame, global_prediction, filename)

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
                                 pred_worker=None,
                                 recording: bool = False,
                                 output_video_path: Optional[str] = None,
                                 has_occlusion: bool = False,
                                 visualize_every: int = 30) -> str:
        """Process the video using discovered seed frames."""

        self.label_registry = {"_background_": 0}
        self._committed_seed_frames.clear()
        self._seed_segment_lookup = {}
        self._seed_frames = self.discover_seed_frames(
            self.video_name, self.video_folder)
        if not self._seed_frames:
            message = ("No label frames found. Please label a frame click save  "
                       "(PNG+JSON are saved together) before running CUTIE.")
            logger.info(message)
            if pred_worker is not None:
                pred_worker.stop_signal.emit()
            return message

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
