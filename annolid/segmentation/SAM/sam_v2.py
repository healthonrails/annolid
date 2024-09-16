import os
# Enable CPU fallback for unsupported MPS ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import glob
import torch
import numpy as np
import cv2

from annolid.utils.files import download_file
from annolid.annotation.label_processor import LabelProcessor
from annolid.gui.shape import MaskShape
from annolid.annotation.keypoints import save_labels
from annolid.utils.devices import get_device
from annolid.utils.videos import extract_frames_with_opencv


class SAM2VideoProcessor:
    def __init__(self, video_dir, id_to_labels,
                 checkpoint_path=None,
                 model_config="sam2_hiera_l.yaml",
                 epsilon_for_polygon=2.0):
        """
        Initializes the SAM2VideoProcessor with the given parameters.

        Args:
            video_dir (str): Directory containing video frames.
            id_to_labels (dict): Mapping of object IDs to labels.
            checkpoint_path (str, optional): Path to the model checkpoint.
            model_config (str): Path to the model configuration file.
            epsilon_for_polygon (float): Epsilon value for polygon approximation.
        """
        # Set default checkpoint path if not provided
        if checkpoint_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(current_dir,
                                           "segment-anything-2",
                                           "checkpoints",
                                           "sam2_hiera_large.pt")

        self.BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"

        self.video_dir = video_dir
        self.checkpoint_path = checkpoint_path
        self.model_config = model_config
        self.id_to_labels = id_to_labels
        self.device = get_device()
        self.epsilon_for_polygon = epsilon_for_polygon
        self.frame_names = self._load_frame_names()
        self.predictor = self._initialize_predictor()
        
        self._handle_device_specific_settings()

    def _initialize_predictor(self):
        """Initializes the SAM2 video predictor."""
        from sam2.build_sam import build_sam2_video_predictor
        if not os.path.exists(self.checkpoint_path):
            sam2_checkpoint_url = f"{self.BASE_URL}{os.path.basename(self.checkpoint_path)}"
            download_file(sam2_checkpoint_url, self.checkpoint_path)

        return build_sam2_video_predictor(self.model_config,
                                          self.checkpoint_path,
                                          device=self.device)

    def _handle_device_specific_settings(self):
        """Handles settings specific to the device (MPS or CUDA)."""
        if self.device == 'mps':
            self._warn_about_mps_support()
        elif self.device == 'cuda' and torch.cuda.is_available():
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

    def _load_frame_names(self):
        """Loads and sorts JPEG frame names from the specified directory."""
        try:
            frame_names = [
                p for p in os.listdir(self.video_dir)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
            ]
            frame_names.sort(key=lambda p: (os.path.splitext(p)[0]))
            return frame_names
        except FileNotFoundError as e:
            print(f"Error loading frames: {e}")
            return []

    def get_frame_shape(self):
        """Returns the shape of the first frame in the video directory."""
        first_frame_path = os.path.join(self.video_dir, self.frame_names[0])
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            raise ValueError(
                f"Unable to read the first frame from {first_frame_path}")
        return first_frame.shape

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
            annot_type = annotation['type']
            if annot_type == 'points':
                self._add_points(inference_state, frame_idx, obj_id,
                                 annotation['points'], annotation['labels'])
            elif annot_type == 'box':
                self._add_box(inference_state, frame_idx,
                              obj_id, annotation['box'])
            elif annot_type == 'mask':
                self._add_mask(inference_state, frame_idx,
                               obj_id, annotation['mask'])
            else:
                print(f"Unknown annotation type: {annot_type}")

    def _add_points(self, inference_state, frame_idx, obj_id, points, labels):
        """Handles the addition of points annotations."""
        self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=np.array(points, dtype=np.float32),
            labels=np.array(labels, dtype=np.int32)
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

    def _save_annotations(self, filename, mask_dict, frame_shape):
        """Saves annotations to a JSON file."""
        height, width, _ = frame_shape
        image_path = os.path.splitext(filename)[0] + '.jpg'
        label_list = []
        for label_id, mask in mask_dict.items():
            label = self.id_to_labels.get(int(label_id), str(label_id))
            current_shape = MaskShape(label=label,
                                      flags={},
                                      description='grounding_sam')
            current_shape.mask = mask
            _shapes = current_shape.toPolygons(
                epsilon=self.epsilon_for_polygon)
            if not _shapes:
                continue
            current_shape = _shapes[0]
            points = [[point.x(), point.y()] for point in current_shape.points]
            current_shape.points = points
            label_list.append(current_shape)
        save_labels(filename=filename, imagePath=image_path, label_list=label_list,
                    height=height, width=width, save_image_to_json=False)
        return label_list

    def _propagate(self, inference_state):
        """Runs mask propagation and visualizes the results every few frames."""
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            mask_dict = {}
            filename = os.path.join(self.video_dir, f'{out_frame_idx:09}.json')
            for i, out_obj_id in enumerate(out_obj_ids):
                _obj_mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                mask_dict[str(out_obj_id)] = _obj_mask
            self._save_annotations(filename, mask_dict, self.frame_shape)

    def run(self, annotations, frame_idx):
        """
        Runs the analysis workflow with specified annotations and frame index.

        Args:
            annotations (list): List of annotation dictionaries, each with 'type', 'points', and 'labels'.
            frame_idx (int): Index of the frame to start the analysis.
        """
        inference_state = self.predictor.init_state(
            video_path=self.video_dir,
            async_loading_frames=True,
        )
        self.predictor.reset_state(inference_state)
        self.frame_shape = self.get_frame_shape()

        # Add annotations and display results
        for annotation in annotations:
            frame_idx = annotation['ann_frame_idx'] if 'ann_frame_idx' in annotation else frame_idx
            self.add_annotations(inference_state, frame_idx,
                                 annotation.get('obj_id', 1), [annotation])

        # Propagate and visualize the results
        self._propagate(inference_state)


def process_video(video_path,
                  frame_idx=0,
                  checkpoint_path=None,
                  model_config="sam2_hiera_l.yaml",
                  epsilon_for_polygon=2.0):
    """
    Processes a video by extracting frames, loading annotations from multiple JSON files, and running analysis.

    Args:
        video_path (str): Path to the video file.
        frame_idx (int): Start from the first frame.
        checkpoint_path (str, optional): Path to the model checkpoint.
        model_config (str, optional): Path to the model configuration file.
        epsilon_for_polygon (float, optional): Epsilon value for polygon approximation.
    """
    # Extract frames from the video
    video_dir = extract_frames_with_opencv(video_path)

    # Find all JSON annotation files in the directory
    anno_jsons = glob.glob(os.path.join(video_dir, "*.json"))
    if not anno_jsons:
        raise FileNotFoundError(
            "No JSON annotation files found in the video directory.")

    id_to_labels = {}
    all_annotations = []

    # Loop through all the JSON annotation files
    for anno_json in anno_jsons:
        # Initialize the LabelProcessor with each JSON file
        ann_frame_idx = os.path.splitext(os.path.basename(anno_json))[0]
        if '_' in ann_frame_idx:
            ann_frame_idx = ann_frame_idx.split('_')[-1]
        try:
            ann_frame_idx = int(ann_frame_idx)
        except ValueError:
            ann_frame_idx = 0
        label_processor = LabelProcessor(anno_json)

        # Convert shapes to the custom annotations format
        annotations = label_processor.convert_shapes_to_annotations(ann_frame_idx)
        all_annotations.extend(annotations)

        # Update the mapping of object IDs to labels
        id_to_labels.update(label_processor.get_id_to_labels())

    # Initialize the analyzer
    analyzer = SAM2VideoProcessor(
        video_dir=video_dir,
        id_to_labels=id_to_labels,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        epsilon_for_polygon=epsilon_for_polygon
    )

    # Run the analysis with the combined annotations
    analyzer.run(all_annotations, frame_idx)


# Example usage
if __name__ == "__main__":

    video_path = os.path.expanduser(
        "~/Downloads/mouse.mp4")
    process_video(video_path)
