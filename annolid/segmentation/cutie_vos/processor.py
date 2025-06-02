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

    def _save_frame_annotation(self, frame_idx: int, predicted_mask_np: np.ndarray,
                               labels_map_id_to_name: Dict[int, str], frame_shape: tuple):
        """Saves the predicted mask for a single frame as a JSON annotation."""
        filename_json = self.video_folder / \
            f"{self.video_path.stem}_{frame_idx:09d}.json"
        height, width, _ = frame_shape

        label_list_for_save: List[Shape] = []
        unique_object_ids_in_pred = np.unique(predicted_mask_np)

        for obj_id in unique_object_ids_in_pred:
            if obj_id == 0:
                continue  # Skip background

            obj_label_str = labels_map_id_to_name.get(
                obj_id, f"obj_{obj_id}")  # Fallback label
            # Boolean mask for this object
            single_obj_mask_bool = (predicted_mask_np == obj_id)

            # Create MaskShape and convert to polygons for saving
            mask_shape_obj = MaskShape(
                label=obj_label_str, flags={}, description=f"cutie_vos_segment")
            mask_shape_obj.mask = single_obj_mask_bool  # Assign boolean mask

            polygon_shapes = mask_shape_obj.toPolygons(
                epsilon=self.config.get('epsilon_for_polygon', 2.0))
            if polygon_shapes:
                # Annolid's save_labels might expect Shape, not MaskShape, or handle both
                # For now, assume we convert to basic Shape with points
                main_polygon = polygon_shapes[0]  # Take the first polygon
                shape_to_save = Shape(label=obj_label_str, shape_type='polygon',
                                      flags=main_polygon.flags, description=main_polygon.description)
                shape_to_save.points = [[p.x(), p.y()]
                                        for p in main_polygon.points]
                label_list_for_save.append(shape_to_save)
            else:
                logger.warning(
                    f"Could not extract polygon for object ID {obj_id}(label: {obj_label_str}) in frame {frame_idx}.")

        if label_list_for_save:
            save_labels(filename=str(filename_json), imagePath=None,  # No image data in JSON
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

        prev_frame_bgr_for_flow: Optional[np.ndarray] = None
        frames_actually_processed_count = 0
        # Default if loop doesn't run
        final_message = f"Segment processed up to frame {self.segment_start_frame -1 }."

        try:

            if self.annotated_frame != self.segment_start_frame:
                logger.warning(f"Segment for {self.video_path.name}: Annotated frame {self.annotated_frame} "
                               f"differs from segment processing start frame {self.segment_start_frame}. "
                               f"Mask from frame {self.annotated_frame} will be applied at frame {self.segment_start_frame}.")

            # Frame where Cutie applies the initial mask & starts reading
            cutie_start_frame_for_mask = self.segment_start_frame

            for frame_idx, pred_mask_np in self.cutie_engine.process_frames(
                video_capture=cap,
                # Cutie starts reading and applies mask here
                start_frame_index=cutie_start_frame_for_mask,
                initial_mask_np=self.initial_mask_info['mask_np'],
                num_objects_in_mask=self.initial_mask_info['num_objects'],
                frames_to_propagate=frames_to_propagate_this_segment,  # Max frames for this segment
                pred_worker=self.pred_worker
            ):
                # frame_idx is the actual video frame number processed by CutieEngine
                # We only care about saving annotations if frame_idx is within our defined segment
                if not (self.segment_start_frame <= frame_idx <= self.segment_end_frame):
                    # This should not happen if CutieEngine starts at segment_start_frame and runs for correct duration
                    logger.warning(f"CutieEngine yielded frame {frame_idx} outside target segment "
                                   f"[{self.segment_start_frame}-{self.segment_end_frame}]. Ignoring.")
                    continue

                self._save_frame_annotation(
                    frame_idx, pred_mask_np,
                    self.initial_mask_info['labels_map_id_to_name'],
                    self.initial_mask_info['frame_shape']
                )
                frames_actually_processed_count += 1

                # Optional: Optical flow and video recording
                if self.video_writer or self.config.get('compute_optical_flow', False):
                    # Re-read frame for visualization/flow
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, current_frame_bgr_for_viz = cap.read()
                    if ret:
                        if self.config.get('compute_optical_flow', False) and prev_frame_bgr_for_flow is not None:
                            flow_hsv, flow_xy = compute_optical_flow(
                                prev_frame_bgr_for_flow, current_frame_bgr_for_viz)
                            # Flow could be used or saved alongside JSON if needed

                        if self.video_writer:
                            overlay = overlay_davis(
                                current_frame_bgr_for_viz, pred_mask_np)  # Uses BGR
                            # Add flow to overlay if computed and desired
                            self.video_writer.write(overlay)
                        prev_frame_bgr_for_flow = current_frame_bgr_for_viz.copy()

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
            # Optionally clear CutieEngine memory if it's reused for more segments by the same worker
            # if self.cutie_engine: self.cutie_engine.clear_memory()

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
