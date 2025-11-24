from annolid.utils.logger import logger
from annolid.utils.files import download_file
from annolid.utils.devices import get_device
from annolid.utils.annotation_store import AnnotationStore
from annolid.gui.shape import MaskShape
from annolid.annotation.label_processor import LabelProcessor
from annolid.annotation.keypoints import save_labels
from annolid.annotation.detections_writer import DetectionsWriter
from typing import Optional
from pathlib import Path
import torch
import numpy as np
import glob
import cv2
import copy
import os
# Enable CPU fallback for unsupported MPS ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class BaseSAMVideoProcessor:
    """Shared utilities for SAM-style video propagation."""

    def __init__(self, video_dir, id_to_labels, epsilon_for_polygon=2.0):
        self.video_path = video_dir  # original path (file or dir)
        self.video_dir = self._normalize_video_path(video_dir)
        self.id_to_labels = id_to_labels or {}
        self.epsilon_for_polygon = epsilon_for_polygon
        self.device = get_device()
        self.frame_names = self._load_frame_names()
        self.frame_shape = None
        self.ndjson_writer: Optional[DetectionsWriter] = None

    def _init_ndjson_writer(self, ndjson_filename: Optional[str]) -> None:
        """Initialise an NDJSON writer if a filename is provided."""
        if not ndjson_filename:
            return
        try:
            self.ndjson_writer = DetectionsWriter(
                Path(self.video_dir),
                enable_annotation_store=False,
                ndjson_filename=ndjson_filename,
            )
        except Exception as exc:
            logger.warning(
                "Unable to set up NDJSON writer at %s: %s",
                self.video_dir,
                exc,
            )
            self.ndjson_writer = None

    @staticmethod
    def _normalize_video_path(path):
        """
        Normalize the video path to a directory. If a file path is provided,
        strip the extension to map to the extracted frames folder.
        """
        if os.path.isfile(path):
            return os.path.splitext(path)[0]
        if os.path.isdir(path):
            return path
        raise ValueError(f"Invalid path: {path}")

    def _load_frame_names(self):
        """Loads and sorts JPEG/PNG frame names from the video directory."""
        try:
            frame_names = [
                p
                for p in os.listdir(self.video_dir)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
            ]
            frame_names.sort(key=lambda p: os.path.splitext(p)[0])
            return frame_names
        except FileNotFoundError:
            return []

    def _video_first_frame_shape(self):
        """Read the first frame directly from the video file if available."""
        if not os.path.isfile(self.video_path):
            return None
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        return frame.shape

    def get_frame_shape(self):
        """Returns the shape of the first frame in the video directory."""
        if self.frame_shape is not None:
            return self.frame_shape

        # Prefer extracted frames if present.
        if self.frame_names:
            first_frame_path = os.path.join(
                self.video_dir, self.frame_names[0])
            first_frame = cv2.imread(first_frame_path)
            if first_frame is None:
                raise ValueError(
                    f"Unable to read the first frame from {first_frame_path}"
                )
            self.frame_shape = first_frame.shape
            return self.frame_shape

        # Fallback: read from the video file directly.
        shape = self._video_first_frame_shape()
        if shape is None:
            raise ValueError(
                f"Unable to determine frame shape from {self.video_dir} or {self.video_path}"
            )
        self.frame_shape = shape
        return self.frame_shape

    def total_frames_estimate(self) -> Optional[int]:
        """Best-effort estimate of total frames for progress reporting."""
        if self.frame_names:
            return len(self.frame_names)
        if os.path.isfile(self.video_path):
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                cap.release()
                return count if count > 0 else None
        return None

    def _save_annotations(
        self,
        filename,
        mask_dict,
        frame_shape,
        frame_idx: Optional[int] = None,
        obj_meta: Optional[dict] = None,
        frame_meta: Optional[dict] = None,
    ):
        """Saves annotations to a JSON file."""
        height, width = frame_shape[:2]
        image_path = os.path.splitext(filename)[0] + ".jpg"
        label_list = []
        ndjson_shapes = []
        meta_lookup = obj_meta or {}

        for label_id, mask in mask_dict.items():
            label = self.id_to_labels.get(int(label_id), str(label_id))
            current_shape = MaskShape(
                label=label, flags={}, description="grounding_sam"
            )
            current_shape.mask = mask
            try:
                current_shape.other_data.update(
                    meta_lookup.get(str(label_id), {}))
            except Exception:
                # Keep best-effort metadata attachment; shapes still persist.
                pass
            polygons = current_shape.toPolygons(
                epsilon=self.epsilon_for_polygon
            )
            if not polygons:
                continue
            current_shape = polygons[0]
            points = [[point.x(), point.y()] for point in current_shape.points]
            current_shape.points = points
            label_list.append(current_shape)
            # Preserve mask only for NDJSON output to avoid bloating the annotation store.
            nd_shape = copy.deepcopy(current_shape)
            nd_shape.mask = mask
            ndjson_shapes.append(nd_shape)
        if label_list:
            save_labels(
                filename=filename,
                imagePath=image_path,
                label_list=label_list,
                height=height,
                width=width,
                save_image_to_json=False,
                persist_json=False,
            )
        # Write an NDJSON record if configured.
        frame_number = (
            frame_idx
            if frame_idx is not None
            else AnnotationStore.frame_number_from_path(Path(filename))
        )
        if self.ndjson_writer and frame_number is not None and ndjson_shapes:
            try:
                self.ndjson_writer.write(
                    frame_number,
                    frame_shape,
                    ndjson_shapes,
                    frame_other_data=frame_meta,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to append NDJSON for frame %s: %s",
                    frame_number,
                    exc,
                )
        return label_list


class SAM2VideoProcessor(BaseSAMVideoProcessor):
    BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"

    def __init__(
        self,
        video_dir,
        id_to_labels,
        checkpoint_path=None,
        model_config="sam2.1_hiera_s.yaml",
        epsilon_for_polygon=2.0,
    ):
        """
        Initializes the SAM2VideoProcessor with the given parameters.

        Args:
            video_dir (str): Directory containing video frames.
            id_to_labels (dict): Mapping of object IDs to labels.
            checkpoint_path (str, optional): Path to the model checkpoint.
            model_config (str): Path to the model configuration file.
            epsilon_for_polygon (float): Epsilon value for polygon approximation.
        """
        self.model_config = model_config
        self.checkpoint_path = self._resolve_checkpoint_path(
            checkpoint_path, model_config
        )
        super().__init__(video_dir, id_to_labels, epsilon_for_polygon)
        self.predictor = self._initialize_predictor()
        self._handle_device_specific_settings()

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path, model_config):
        """Resolve a checkpoint path or fall back to the bundled default."""
        if checkpoint_path is not None:
            return checkpoint_path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint = (
            "sam2.1_hiera_small.pt"
            if "hiera_s" in model_config
            else "sam2.1_hiera_large.pt"
        )
        return os.path.join(
            current_dir, "segment-anything-2", "checkpoints", checkpoint
        )

    def _initialize_predictor(self):
        """Initializes the SAM2 video predictor."""
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            print(
                "The segment-anything-2 package is not installed. "
                "Please install it with the following commands:"
            )
            print(
                "cd /path/to/annolid/annolid/segmentation/SAM/segment-anything-2"
            )
            print("pip install -e .")
            raise
        if not os.path.exists(self.checkpoint_path):
            sam2_checkpoint_url = (
                f"{self.BASE_URL}{os.path.basename(self.checkpoint_path)}"
            )
            download_file(sam2_checkpoint_url, self.checkpoint_path)

        print(self.model_config, self.checkpoint_path)
        return build_sam2_video_predictor(
            self.model_config, self.checkpoint_path, device=self.device
        )

    def _handle_device_specific_settings(self):
        """Handles settings specific to the device (MPS or CUDA)."""
        if self.device == "mps":
            self._warn_about_mps_support()
        elif self.device == "cuda" and torch.cuda.is_available():
            self._enable_cuda_optimizations()

    def _warn_about_mps_support(self):
        """Prints a warning about preliminary support for MPS devices."""
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    def _enable_cuda_optimizations(self):
        """Enables CUDA-specific optimizations for compatible devices."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def add_annotations(self, inference_state, frame_idx, obj_id, annotations):
        """
        Adds annotations to the predictor and updates the mask.

        Args:
            inference_state: The current inference state of the predictor.
            frame_idx (int): Index of the frame to annotate.
            obj_id (int): Object ID for the annotations.
            annotations (list): List of annotation dictionaries,
            each with 'type', 'points', and 'labels'.
        """
        for annotation in annotations:
            annot_type = annotation["type"]
            if annot_type == "points":
                self._add_points(
                    inference_state,
                    frame_idx,
                    obj_id,
                    annotation["points"],
                    annotation["labels"],
                )
            elif annot_type == "box":
                self._add_box(
                    inference_state, frame_idx, obj_id, annotation["box"]
                )
            elif annot_type == "mask":
                self._add_mask(
                    inference_state, frame_idx, obj_id, annotation["mask"]
                )
            else:
                print(f"Unknown annotation type: {annot_type}")

    def _add_points(self, inference_state, frame_idx, obj_id, points, labels):
        """Handles the addition of points annotations."""
        self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=np.array(points, dtype=np.float32),
            labels=np.array(labels, dtype=np.int32),
        )

    def _add_box(self, inference_state, frame_idx, obj_id, box):
        """Handles the addition of box annotations."""
        self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box,
        )

    def _add_mask(self, inference_state, frame_idx, obj_id, mask):
        """Handles the addition of mask annotations."""
        self.predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=mask,
        )

    def _propagate(self, inference_state):
        """Runs mask propagation and visualizes the results every few frames."""
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(inference_state):
            mask_dict = {}
            filename = os.path.join(
                self.video_dir, f"{out_frame_idx:09}.json"
            )
            for i, out_obj_id in enumerate(out_obj_ids):
                _obj_mask = (
                    (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                )
                mask_dict[str(out_obj_id)] = _obj_mask
            self._save_annotations(
                filename, mask_dict, self.frame_shape, frame_idx=out_frame_idx
            )

    def run(self, annotations, frame_idx):
        """
        Runs the analysis workflow with specified annotations and frame index.

        Args:
            annotations (list): List of annotation dictionaries, each with 'type', 'points', and 'labels'.
            frame_idx (int): Index of the frame to start the analysis.
        """
        inference_state = self.predictor.init_state(
            video_path=self.video_path,
            async_loading_frames=True,
        )
        self.predictor.reset_state(inference_state)
        self.frame_shape = self.get_frame_shape()

        for annotation in annotations:
            frame_idx = (
                annotation["ann_frame_idx"]
                if "ann_frame_idx" in annotation
                else frame_idx
            )
            self.add_annotations(
                inference_state,
                frame_idx,
                annotation.get("obj_id", 1),
                [annotation],
            )

        self._propagate(inference_state)


def load_annotations_from_video(video_path):
    """
    Load LabelMe annotations that sit next to extracted frames.

    Returns a tuple of (annotations, id_to_labels).
    """
    video_dir = BaseSAMVideoProcessor._normalize_video_path(video_path)
    anno_jsons = sorted(glob.glob(os.path.join(video_dir, "*.json")))
    if not anno_jsons:
        raise FileNotFoundError(
            "No JSON annotation files found in the video directory."
        )

    id_to_labels = {}
    all_annotations = []

    for anno_json in anno_jsons:
        ann_frame_idx = os.path.splitext(os.path.basename(anno_json))[0]
        if "_" in ann_frame_idx:
            ann_frame_idx = ann_frame_idx.split("_")[-1]
        try:
            ann_frame_idx = int(ann_frame_idx)
        except ValueError:
            ann_frame_idx = 0
        label_processor = LabelProcessor(anno_json)
        annotations = label_processor.convert_shapes_to_annotations(
            ann_frame_idx
        )
        all_annotations.extend(annotations)
        id_to_labels.update(label_processor.get_id_to_labels())

    return all_annotations, id_to_labels


def process_video(
    video_path,
    frame_idx=0,
    checkpoint_path=None,
    model_config="sam2.1_hiera_s.yaml",
    epsilon_for_polygon=2.0,
):
    """
    Processes a video by extracting frames, loading annotations from multiple JSON files, and running analysis.

    Args:
        video_path (str): Path to the video file.
        frame_idx (int): Start from the first frame.
        checkpoint_path (str, optional): Path to the model checkpoint.
        model_config (str, optional): Path to the model configuration file.
        epsilon_for_polygon (float, optional): Epsilon value for polygon approximation.
    """
    all_annotations, id_to_labels = load_annotations_from_video(video_path)

    analyzer = SAM2VideoProcessor(
        video_dir=video_path,
        id_to_labels=id_to_labels,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        epsilon_for_polygon=epsilon_for_polygon,
    )

    try:
        analyzer.run(all_annotations, frame_idx)
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    video_path = os.path.expanduser("~/Downloads/mouse.mp4")
    process_video(video_path)
