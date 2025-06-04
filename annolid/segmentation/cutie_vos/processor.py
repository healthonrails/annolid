import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import torch
from .engine import CutieEngine

# Annolid-specific utilities
from annolid.data.videos import CV2Video as AnnolidCV2VideoUtil
from annolid.gui.label_file import LabelFile
from annolid.gui.shape import MaskShape, Shape  # For saving annotations
from annolid.annotation.keypoints import save_labels  # For saving annotations
from labelme.utils.shape import shapes_to_label
from annolid.utils.logger import logger

# Optional: For flow and visualization if kept here
from annolid.motion.optical_flow import compute_optical_flow
from annolid.utils import draw  # For drawing flow
# For visualization video
from annolid.segmentation.cutie_vos.interactive_utils import overlay_davis
from annolid.segmentation.cutie_vos.predict import find_mask_center_opencv
from annolid.utils.files import create_tracking_csv_file


class SegmentedCutieExecutor:
    """
    Orchestrates Cutie VOS for a single, defined tracking segment using CutieEngine.
    It handles loading initial masks, interfacing with CutieEngine,
    and saving Annolid-specific annotations.
    """

    def __init__(self,
                 video_path_str: str,
                 segment_annotated_frame: int,
                 segment_start_frame: int,
                 segment_end_frame: int,
                 processing_config: Dict,  # Contains Cutie engine configs and other executor configs
                 pred_worker: Optional[object] = None,
                 device: Optional[torch.device] = None):  # Pass device explicitly

        self.video_path = Path(video_path_str)
        self.video_folder = self.video_path.with_suffix('')

        self.annotated_frame = segment_annotated_frame
        self.segment_start_frame = segment_start_frame
        self.segment_end_frame = segment_end_frame

        self.config = processing_config  # Store the full config
        self.pred_worker = pred_worker
        self.device = device  # Store the device for CutieEngine

        self.cutie_engine: Optional[CutieEngine] = None
        # {'mask': np.array, 'labels_dict': dict, 'num_objects': int}
        self.initial_mask_info: Optional[Dict] = None
        self.video_writer: Optional[cv2.VideoWriter] = None

        # For _save_per_object_results
        self._current_frame_idx_for_saving: Optional[int] = None
        # List of dicts like {'frame_number':idx, 'instance_name':str, 'cx':float, 'cy':float, 'motion_index':float}
        # ** NEW: Initialize lists for CSV data accumulation for this segment **
        self._csv_frame_numbers: List[int] = []
        self._csv_instance_labels: List[str] = []
        self._csv_cx_values: List[float] = []  # Use float for precision
        self._csv_cy_values: List[float] = []
        self._csv_motion_indices: List[float] = []

        self._current_frame_idx_for_saving: Optional[int] = None

        self._initialize_engine()

    def _initialize_engine(self):
        """Initializes the CutieEngine."""
        logger.info(
            f"SegmentedCutieExecutor: Initializing CutieEngine for video {self.video_path.name}")
        logger.info(
            f"segment {self.segment_start_frame}-{self.segment_end_frame}")
        try:
            # Extract configs specifically for CutieEngine from self.config
            cutie_engine_config_overrides = {
                'mem_every': self.config.get('mem_every', 5),
                'max_mem_frames': self.config.get('t_max_value', 5),
                # Add other overrides if CutieEngine's Hydra config supports them
            }
            self.cutie_engine = CutieEngine(
                cutie_config_overrides=cutie_engine_config_overrides,
                device=self.device
                # model_weights_path can be passed from self.config if needed
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize CutieEngine for {self.video_path.name}: {e}", exc_info=True)
            self.cutie_engine = None

    def _prepare_initial_mask(self) -> bool:
        """
        Loads annotation for self.annotated_frame, prepares initial mask and label mapping.
        Stores result in self.initial_mask_info. Returns True if successful.
        """
        annotation_json_path = self.video_folder / \
            f"{self.video_path.stem}_{self.annotated_frame:09d}.json"
        if not annotation_json_path.exists():
            logger.error(f"Annotation file {annotation_json_path.name}")
            logger.error(
                f"for segment(annotated at {self.annotated_frame}) not found.")
            return False

        try:
            label_file = LabelFile(
                str(annotation_json_path), is_video_frame=True)
            shapes_from_json = label_file.shapes

            valid_shapes = [
                s for s in shapes_from_json
                if s.get('shape_type') == 'polygon' and
                len(s.get('points', [])) >= 3 and
                'zone' not in ((s.get('description', '') or '').lower(
                ) + (s.get('label', '') or '').lower())
            ]
            if not valid_shapes:
                logger.warning(
                    f"No valid polygons found in {annotation_json_path.name}. Cannot create initial mask.")
                return False

            # Create labels_dict (numeric IDs for Cutie) and value_to_name (for saving results)
            # Background is 0. Objects start from 1.
            current_id = 1
            labels_dict_numeric_ids = {}  # e.g. {'animal_1': 1, 'animal_2': 2}
            # This is what shapes_to_label expects for label_name_to_value

            for shape in sorted(valid_shapes, key=lambda x: x["label"]):
                label_name = shape["label"]
                if label_name not in labels_dict_numeric_ids:
                    labels_dict_numeric_ids[label_name] = current_id
                    current_id += 1

            num_objects = len(labels_dict_numeric_ids)
            if num_objects == 0:
                logger.warning("No objects found after filtering shapes.")
                return False

            # Use Annolid's CV2Video util to get frame dimensions safely
            temp_video_loader = AnnolidCV2VideoUtil(str(self.video_path))
            annotated_frame_image = temp_video_loader.load_frame(
                self.annotated_frame)
            del temp_video_loader
            if annotated_frame_image is None:
                logger.error(
                    f"Could not load annotated frame {self.annotated_frame} to get dimensions.")
                return False

            image_height, image_width, _ = annotated_frame_image.shape

            # Create the initial NumPy mask with object IDs
            initial_mask_np, _ = shapes_to_label(
                (image_height, image_width), valid_shapes, labels_dict_numeric_ids)
            if initial_mask_np is None:
                logger.error(
                    f"Failed to generate initial mask from {annotation_json_path.name}.")
                return False

            self.initial_mask_info = {
                'mask_np': initial_mask_np,  # (H, W) with object IDs 1, 2, ...
                # {1: 'animal_1', ...}
                'labels_map_id_to_name': {v: k for k, v in labels_dict_numeric_ids.items()},
                'num_objects': num_objects,
                # Store for saving annotations
                'frame_shape': (image_height, image_width, 3)
            }
            logger.info(
                f"Prepared initial mask for segment(annotated at {self.annotated_frame}) with {num_objects} objects.")
            return True
        except Exception as e:
            logger.error(
                f"Error preparing initial mask from {annotation_json_path.name}: {e}", exc_info=True)
            return False

    def _initialize_segment_video_writer(self, fps: float):
        """Initializes video writer if recording is enabled for this segment."""
        if self.config.get('save_video_with_color_mask', False) and self.initial_mask_info:
            output_video_path_str = str(self.video_folder / f"{self.video_path.stem}_segment_{self.segment_start_frame}_to_{self.segment_end_frame}_overlay.mp4"
                                        )
            h, w, _ = self.initial_mask_info['frame_shape']
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_video_path_str, fourcc, fps, (w, h))
            logger.info(
                f"Recording segment overlay to: {output_video_path_str}")

    # To calculate motion index and store per-object data for the current frame
    def _process_object_metrics(self,
                                object_label_str: str,
                                object_mask_np: np.ndarray,  # Boolean mask for this single object
                                dense_flow_xy: Optional[np.ndarray]):  # Optical flow (dx, dy) for the current frame transition
        """Calculates centroid and motion index for a single object mask."""

        motion_index = -1.0
        cx, cy = find_mask_center_opencv(object_mask_np)

        if dense_flow_xy is not None and self.config.get('compute_optical_flow', True):
            magnitude = np.sqrt(
                dense_flow_xy[..., 0]**2 + dense_flow_xy[..., 1]**2)
            sum_mask = np.sum(object_mask_np)  # Sum of boolean mask
            if sum_mask > 0:
                motion_index = np.sum(object_mask_np * magnitude) / sum_mask
            else:
                motion_index = 0.0  # No motion if mask is empty

        # Store for potential CSV or detailed logging
        if self._current_frame_idx_for_saving is not None:  # Ensure frame context is set
            self._csv_frame_numbers.append(self._current_frame_idx_for_saving)
            self._csv_instance_labels.append(object_label_str)
            self._csv_cx_values.append(cx)
            self._csv_cy_values.append(cy)
            self._csv_motion_indices.append(motion_index)

        return motion_index  # Return for direct use in description

    def _save_frame_annotation(self,
                               frame_idx: int,
                               predicted_mask_np_object_ids: np.ndarray,  # Mask with numeric object IDs
                               labels_map_id_to_name: Dict[int, str],
                               frame_shape: tuple,
                               dense_flow_for_this_frame: Optional[np.ndarray]):  # Pass the computed flow
        """Saves the predicted mask for a single frame as a JSON annotation, including motion index."""
        self._current_frame_idx_for_saving = frame_idx  # Set for _process_object_metrics

        filename_json = self.video_folder / \
            f"{self.video_path.stem}_{frame_idx:09d}.json"
        height, width, _ = frame_shape
        label_list_for_save: List[Shape] = []
        unique_object_ids_in_pred = np.unique(predicted_mask_np_object_ids)

        for obj_id in unique_object_ids_in_pred:
            if obj_id == 0:
                continue

            obj_label_str = labels_map_id_to_name.get(obj_id, f"obj_{obj_id}")
            single_obj_mask_bool = (predicted_mask_np_object_ids == obj_id)

            # Calculate motion index for this specific object's mask
            motion_idx_for_this_object = self._process_object_metrics(
                obj_label_str, single_obj_mask_bool, dense_flow_for_this_frame
            )

            mask_shape_obj = MaskShape(
                label=obj_label_str, flags={},
                # Include motion index
                description=f'cutie_vos_segment; motion_index: {motion_idx_for_this_object:.2f}'
            )
            mask_shape_obj.mask = single_obj_mask_bool

            polygon_shapes = mask_shape_obj.toPolygons(
                epsilon=self.config.get('epsilon_for_polygon', 2.0))
            if polygon_shapes:
                main_polygon = polygon_shapes[0]
                shape_to_save = Shape(label=obj_label_str, shape_type='polygon',
                                      flags=main_polygon.flags, description=main_polygon.description)
                shape_to_save.points = [[p.x(), p.y()]
                                        for p in main_polygon.points]
                label_list_for_save.append(shape_to_save)
            else:
                logger.warning(
                    f"Could not extract polygon for ID {obj_id} (label: {obj_label_str}) in frame {frame_idx}.")

        if label_list_for_save:
            save_labels(filename=str(filename_json), imagePath=None,
                        label_list=label_list_for_save,
                        height=height, width=width, save_image_to_json=False)

    def process_segment(self) -> str:
        """
        Processes the defined video segment using CutieEngine and saves Annolid annotations.
        Returns a status message.
        """
        if not self.cutie_engine:
            return "Error: CutieEngine not initialized."
        if not self._prepare_initial_mask() or not self.initial_mask_info:
            return "Error: Failed to prepare initial mask for the segment."

        frames_to_propagate_this_segment = self.segment_end_frame - \
            self.segment_start_frame + 1
        if frames_to_propagate_this_segment <= 0:
            return f"Segment duration is non-positive ({frames_to_propagate_this_segment} frames). Skipping."

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return f"Error: Cannot open video {self.video_path.name} for segment processing."

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        self._initialize_segment_video_writer(
            video_fps)  # Init writer if recording

        self._csv_frame_numbers.clear()
        self._csv_instance_labels.clear()
        self._csv_cx_values.clear()
        self._csv_cy_values.clear()
        self._csv_motion_indices.clear()

        prev_frame_bgr: Optional[np.ndarray] = None
        # Load the frame *before* the segment_start_frame if flow is needed for the very first segment frame
        if self.config.get('compute_optical_flow', True) and self.segment_start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.segment_start_frame - 1)
            ret_prev, prev_frame_bgr = cap.read()
            if not ret_prev:
                prev_frame_bgr = None  # Could not read frame before segment start
        frames_actually_processed_count = 0
        # Default if loop doesn't run
        final_message = f"Segment processed up to frame {self.segment_start_frame -1 }."

        try:
            # Engine reads from this frame
            start_frame_for_cutie_engine = self.segment_start_frame

            for frame_idx, pred_mask_np_obj_ids in self.cutie_engine.process_frames(
                video_capture=cap,
                start_frame_index=start_frame_for_cutie_engine,
                initial_mask_np=self.initial_mask_info['mask_np'],
                num_objects_in_mask=self.initial_mask_info['num_objects'],
                frames_to_propagate=frames_to_propagate_this_segment,
                pred_worker=self.pred_worker
            ):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret_curr, current_frame_bgr = cap.read()
                if not ret_curr:
                    logger.warning(
                        f"Could not read current frame {frame_idx} for flow/viz. Skipping metrics for this frame.")
                    prev_frame_bgr = None  # Cannot compute flow for next iteration
                    continue  # Skip to next prediction from engine

                dense_flow_for_this_frame: Optional[np.ndarray] = None
                if self.config.get('compute_optical_flow', True) and prev_frame_bgr is not None:
                    _, dense_flow_for_this_frame = compute_optical_flow(
                        prev_frame_bgr, current_frame_bgr)

                self._save_frame_annotation(
                    frame_idx, pred_mask_np_obj_ids,
                    self.initial_mask_info['labels_map_id_to_name'],
                    self.initial_mask_info['frame_shape'],
                    dense_flow_for_this_frame  # Pass computed flow
                )
                frames_actually_processed_count += 1
                prev_frame_bgr = current_frame_bgr.copy()

                if self.video_writer:
                    overlay = overlay_davis(
                        current_frame_bgr, pred_mask_np_obj_ids)
                    if dense_flow_for_this_frame is not None and self.config.get('compute_optical_flow', False):
                        mask_for_flow_viz = pred_mask_np_obj_ids > 0
                        overlay_with_flow = draw.draw_flow(overlay.copy(
                        ), dense_flow_for_this_frame * np.expand_dims(mask_for_flow_viz, axis=-1))
                        self.video_writer.write(overlay_with_flow)
                    else:
                        self.video_writer.write(overlay)

                # Current becomes previous for next iteration
                prev_frame_bgr = current_frame_bgr.copy()
                final_message = f"Segment processed up to frame {frame_idx}."

            final_message = (f"Segment processing completed. Processed {frames_actually_processed_count} frames "
                             f"in range [{self.segment_start_frame}-{self.segment_end_frame}].")
        except Exception as e:
            logger.error(
                f"Exception during segment processing for {self.video_path.name}: {e}", exc_info=True)
            final_message = f"Error during segment processing: {str(e)}"
        finally:
            if cap.isOpened():
                cap.release()
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            if self._csv_frame_numbers:  # If any data was collected
                csv_filename = self.video_folder.parent / \
                    f"{self.video_path.stem}_segment_{self.segment_start_frame}_to_{self.segment_end_frame}_tracked.csv"
                try:
                    create_tracking_csv_file(
                        frame_numbers=self._csv_frame_numbers,
                        instance_names=self._csv_instance_labels,
                        cx_values=self._csv_cx_values,
                        cy_values=self._csv_cy_values,
                        motion_indices=self._csv_motion_indices,  # Pass the motion indices
                        output_file=str(csv_filename),
                        video_path=str(self.video_path),
                    )
                    logger.info(
                        f"Saved tracking CSV for segment to: {csv_filename}")
                except Exception as e:
                    logger.error(
                        f"Failed to save tracking CSV for segment {csv_filename.name}: {e}", exc_info=True)
                    final_message += f" (CSV Save Error: {e})"
        return final_message

    def cleanup(self):
        """Releases resources, particularly the CutieEngine."""
        logger.debug(
            f"SegmentedCutieExecutor: Cleaning up for {self.video_path.name}")
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        if self.cutie_engine:
            self.cutie_engine.cleanup()  # Ask engine to clean its resources
            self.cutie_engine = None
