import os
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
from PIL import Image
from annolid.gui.shape import MaskShape
from annolid.annotation.keypoints import save_labels
from annolid.segmentation.cutie_vos.interactive_utils import (
    image_to_torch,
    torch_prob_to_numpy_mask,
    index_numpy_to_one_hot_torch,
    overlay_davis
)
from omegaconf import open_dict
from hydra import compose, initialize
from annolid.segmentation.cutie_vos.model.cutie import CUTIE
from annolid.segmentation.cutie_vos.inference.inference_core import InferenceCore
from pathlib import Path
import gdown
from annolid.utils.devices import get_device
from labelme.logger import logger
from annolid.motion.optical_flow import compute_optical_flow
from annolid.utils import draw
from annolid.utils.files import create_tracking_csv_file

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
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return cx, cy


class CutieVideoProcessor:

    _REMOTE_MODEL_URL = "https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth"
    _MD5 = "a6071de6136982e396851903ab4c083a"

    def __init__(self, video_name, mem_every=5, debug=False):
        self.video_name = video_name
        self.video_folder = Path(video_name).with_suffix("")
        self.mem_every = mem_every
        self.debug = debug
        self.processor = None
        self.num_tracking_instances = 0
        current_file_path = os.path.abspath(__file__)
        self.current_folder = os.path.dirname(current_file_path)
        self.device = get_device()
        self.cutie, self.cfg = self._initialize_model()

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
            cfg['mem_every'] = self.mem_every
            logger.info(
                f"Saving into working memeory for every: {self.mem_every}.")
            cutie_model = CUTIE(cfg).to(self.device).eval()
            model_weights = torch.load(
                cfg.weights, map_location=self.device)
            cutie_model.load_weights(model_weights)
        return cutie_model, cfg

    def _save_annotation(self, filename, mask_dict, frame_shape):
        height, width, _ = frame_shape
        label_list = []
        for label_id, mask in mask_dict.items():
            label = str(label_id)
            self._instance_names.append(label)
            self._frame_numbers.append(self._frame_number)
            cx, cy = find_mask_center_opencv(mask)
            self._cx_values.append(cx)
            self._cy_values.append(cy)
            if self._flow_hsv is not None:
                # unnormalized magnitude
                magnitude = self._flow_hsv[..., 2]
                self._motion_index = np.sum(
                    mask * magnitude) / np.sum(mask)
            else:
                self._motion_index = -1
            self._motion_indices.append(self._motion_index)
            current_shape = MaskShape(label=label,
                                      flags={},
                                      description='grounding_sam')
            current_shape.mask = mask
            current_shape = current_shape.toPolygons()[0]
            points = [[point.x(), point.y()] for point in current_shape.points]
            current_shape.points = points
            label_list.append(current_shape)
        save_labels(filename=filename, imagePath=None, label_list=label_list,
                    height=height, width=width, save_image_to_json=False)

    def process_video_with_mask(self, frame_number=0,
                                mask=None,
                                frames_to_propagate=60,
                                visualize_every=30,
                                labels_dict=None,
                                pred_worker=None,
                                recording=False,
                                output_video_path=None,
                                has_occlusion=False,
                                ):
        if mask is not None:
            num_objects = len(np.unique(mask)) - 1
            self.num_tracking_instances = num_objects
        self.processor = InferenceCore(self.cutie, cfg=self.cfg)
        cap = cv2.VideoCapture(self.video_name)
        value_to_label_names = {
            v: k for k, v in labels_dict.items()} if labels_dict else {}
        instance_names = set(labels_dict.keys())
        if '_background_' in instance_names:
            instance_names.remove('_background_')
        # Get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_number == total_frames - 1:
            return

        # Get the frames per second (fps) of the video
        fps = cap.get(cv2.CAP_PROP_FPS)

        current_frame_index = frame_number
        end_frame_number = frame_number + frames_to_propagate
        current_frame_index = frame_number
        prev_frame = None

        delimiter = '#'
        self.output_tracking_csvpath = str(
            self.video_folder) + f"_start_frame_{current_frame_index}_tracked.csv"

        if recording:
            if output_video_path is None:
                output_video_path = self.output_tracking_csvpath.replace(
                    '.csv', '.mp4')
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.initialize_video_writer(
                output_video_path, frame_width, frame_height, fps)

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=self.device == 'cuda'):
                while cap.isOpened():
                    while not pred_worker.is_stopped():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
                        _, frame = cap.read()
                        if frame is None or current_frame_index > end_frame_number:
                            break
                        self._frame_number = current_frame_index
                        frame_torch = image_to_torch(frame, device=self.device)
                        if (current_frame_index == 0 or
                            (current_frame_index == frame_number == 1) or
                                (frame_number > 1 and
                                 current_frame_index % frame_number == 0)):
                            mask_torch = index_numpy_to_one_hot_torch(
                                mask, num_objects + 1).to(self.device)
                            prediction = self.processor.step(
                                frame_torch, mask_torch[1:], idx_mask=False)
                        else:
                            prediction = self.processor.step(frame_torch)
                        prediction = torch_prob_to_numpy_mask(prediction)

                        if prev_frame is not None:
                            self._flow_hsv, self._flow = compute_optical_flow(
                                prev_frame, frame)
                            self._mask = prediction > 0

                        filename = self.video_folder / \
                            (self.video_folder.name +
                             f"_{current_frame_index:0>{9}}.json")
                        mask_dict = {value_to_label_names.get(label_id, str(label_id)): (prediction == label_id)
                                     for label_id in np.unique(prediction)[1:]}
                        self._save_annotation(filename, mask_dict, frame.shape)
                        # if we lost tracking one of the instances, return the current frame number
                        num_instances_in_current_frame = mask_dict.keys()
                        if len(num_instances_in_current_frame) < self.num_tracking_instances:
                            missing_instances = instance_names - \
                                set(num_instances_in_current_frame)
                            num_missing_instances = self.num_tracking_instances - \
                                len(num_instances_in_current_frame)
                            message = (
                                f"There are {num_missing_instances} missing instance(s) in the current frame ({current_frame_index}).\n\n"
                                f"Here is the list of instances missing or occluded in the current frame:\n"
                                f"Some occluded instances will be recovered automatically in the later frame:\n"
                                f"{', '.join(str(instance) for instance in missing_instances)}"
                            )
                            message_with_index = message + \
                                delimiter + str(current_frame_index)
                            logger.info(message)
                            # Stop the prediction if more than half of the instances are missing,
                            # or when there is no occlusion in the video and one instance loses tracking.
                            if (not has_occlusion or
                                    len(num_instances_in_current_frame) < self.num_tracking_instances / 2
                                    ):
                                pred_worker.stop_signal.emit()
                                # Release the video capture object
                                cap.release()
                                # Release the video writer if recording is set to True
                                if recording:
                                    self.video_writer.release()
                                return message_with_index
                        if recording:
                            visualization = overlay_davis(frame, prediction)
                            if self._flow_hsv is not None:
                                flow_bgr = cv2.cvtColor(
                                    self._flow_hsv, cv2.COLOR_HSV2BGR)
                                expanded_prediction = np.expand_dims(
                                    self._mask, axis=-1)
                                flow_bgr = flow_bgr * expanded_prediction
                                # Reshape mask_array to match the shape of flow_array
                                mask_array_reshaped = np.repeat(
                                    self._mask[:, :, np.newaxis], 2, axis=2)
                                # Overlay optical flow on the frame
                                visualization = cv2.addWeighted(
                                    visualization, 1, flow_bgr, 0.5, 0)
                                visualization = draw.draw_flow(
                                    visualization, self._flow * mask_array_reshaped)

                            # Write the frame to the video file
                            self.video_writer.write(visualization)

                        if self.debug and current_frame_index % visualize_every == 0:
                            visualization = overlay_davis(frame, prediction)
                            plt.imshow(
                                cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
                            plt.title(str(current_frame_index))
                            plt.axis('off')
                            plt.show()
                        # Update prev_frame with the current frame
                        prev_frame = frame.copy()
                        current_frame_index += 1

                    break

                create_tracking_csv_file(self._frame_numbers,
                                         self._instance_names,
                                         self._cx_values,
                                         self._cy_values,
                                         self._motion_indices,
                                         self.output_tracking_csvpath,
                                         fps)
                message = ("Stop at frame:\n") + \
                    delimiter + str(current_frame_index-1)
                pred_worker.stop_signal.emit()
                # Release the video capture object
                cap.release()
                # Release the video writer if recording is set to True
                if recording:
                    self.video_writer.release()
                return message


if __name__ == '__main__':
    # Example usage:
    video_name = 'demo/video.mp4'
    mask_name = 'demo/first_frame.png'
    video_folder = video_name.split('.')[0]
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    mask = np.array(Image.open(mask_name))
    labels_dict = {1: 'object_1', 2: 'object_2',
                   3: 'object_3'}  # Example labels dictionary
    processor = CutieVideoProcessor(video_name, debug=True)
    processor.process_video_with_mask(
        mask=mask, visualize_every=30, frames_to_propagate=30, labels_dict=labels_dict)
