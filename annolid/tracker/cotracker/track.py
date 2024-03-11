import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np
import json
from pathlib import Path
from annolid.annotation.keypoints import save_labels
from annolid.gui.shape import Shape
from annolid.tracker.cotracker.visualizer import Visualizer
"""
@article{karaev2023cotracker,
  title={CoTracker: It is Better to Track Together},
  author={Nikita Karaev and Ignacio Rocco and Benjamin Graham 
  and Natalia Neverova and Andrea Vedaldi and Christian Rupprecht},
  journal={arXiv:2307.07635},
  year={2023}
}
"""


class CoTrackerProcessor:
    def __init__(self, video_path, json_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.video_path = video_path
        self.video_result_folder = Path(self.video_path).with_suffix('')
        if not self.video_result_folder.exists():
            self.video_result_folder.mkdir(exist_ok=True)
        self.point_labels = []
        self.queries = self.load_queries(json_path)
        self.video_height = None
        self.video_width = None

    def get_frame_number(self, json_file):
        # assume json file name pattern as
        # xxxx_000000000.json
        # Split the file name by '_' and '.'
        parts = json_file.split('_')
        # Extract the part between '_' and '.json'
        frame_number_str = parts[-1].split('.')[0]
        # Convert the frame number string to an integer
        frame_number = int(frame_number_str)
        return frame_number

    def load_model(self):
        return torch.hub.load("facebookresearch/co-tracker",
                              "cotracker2_online").to(self.device)

    def load_queries(self, json_path):
        if json_path is None:
            return None

        with open(json_path, 'r') as file:
            data = json.load(file)

        frame_number = self.get_frame_number(json_path)

        queries = []
        for shape in data['shapes']:
            if shape['shape_type'] == 'point':
                label = shape['label']
                # Assuming there's only one point per shape
                points = shape['points'][0]
                self.point_labels.append(label)
                queries.append([frame_number] + points)

        queries_tensor = torch.tensor(queries)
        if torch.cuda.is_available():
            queries_tensor = queries_tensor.cuda()
        return queries_tensor

    def process_step(self,
                     window_frames,
                     is_first_step,
                     grid_size,
                     grid_query_frame):
        video_chunk = torch.tensor(np.stack(
            window_frames[-self.model.step * 2:]),
            device=self.device).float().permute(0, 3, 1, 2)[None]
        if self.queries is not None:
            return self.model(video_chunk,
                              is_first_step=is_first_step,
                              grid_size=grid_size,
                              grid_query_frame=grid_query_frame,
                              queries=self.queries[None])
        else:
            return self.model(video_chunk,
                              is_first_step=is_first_step,
                              grid_size=grid_size,
                              grid_query_frame=grid_query_frame)

    def process_video(self, grid_size=10, grid_query_frame=0):
        if not os.path.isfile(self.video_path):
            raise ValueError("Video file does not exist")

        window_frames = []
        is_first_step = True

        for i, frame in enumerate(iio.imiter(self.video_path, plugin="FFMPEG")):
            if self.video_height is None or self.video_width is None:
                self.video_height, self.video_width, _ = frame.shape
            if i % self.model.step == 0 and i != 0:
                pred_tracks, pred_visibility = self.process_step(
                    window_frames, is_first_step, grid_size, grid_query_frame)
                if pred_tracks is not None:
                    print(i, pred_tracks.shape, pred_visibility.shape)
                is_first_step = False
            window_frames.append(frame)

        pred_tracks, pred_visibility = self.process_step(
            window_frames[-(i % self.model.step) - self.model.step - 1:],
            is_first_step, grid_size, grid_query_frame)

        print("Tracks are computed")

        video = torch.tensor(np.stack(window_frames),
                             device=self.device).permute(0, 3, 1, 2)[None]
        vis = Visualizer(save_dir="./saved_videos", pad_value=120,
                         linewidth=3, tracks_leave_trace=-1)
        res_video, frame_points_dict = vis.visualize(video, pred_tracks, pred_visibility,
                                                     query_frame=grid_query_frame)

        self.save_tracked_points_to_label_json(frame_points_dict)
        return f"All Frames have been processed#{len(frame_points_dict.keys())}"

    def save_results_to_json(self, pred_tracks, pred_visibility, frame_index):
        label_list = []
        for label, point_vis in zip(self.point_labels, pred_tracks):
            point, visible = point_vis
            point = point.tolist()
            cur_shape = Shape(
                label=label,
                flags={},
                description="Cotracker",
                shape_type='point',
                visible=visible
            )
            cur_shape.points = [point]
            label_list.append(cur_shape)

        json_file_path = self.video_result_folder / \
            (self.video_result_folder.name +
             f"_{frame_index:0>{9}}.json")
        save_labels(json_file_path, imagePath="", label_list=label_list,
                    width=self.video_width, height=self.video_height)

    def save_tracked_points_to_label_json(self, frame_points_dict):
        for frame_number, _points in frame_points_dict.items():
            json_file_path = self.video_result_folder / \
                (self.video_result_folder.name +
                 f"_{frame_number:0>{9}}.json")
            label_list = []
            for label, _point in zip(self.point_labels, _points):
                _point, visible = _point
                _point = _point.tolist()
                cur_shape = Shape(
                    label=label,
                    flags={},
                    description="Cotracker",
                    shape_type='point',
                    visible=visible
                )
                cur_shape.points = [_point]
                label_list.append(cur_shape)
            save_labels(json_file_path, imagePath="", label_list=label_list,
                        width=self.video_width, height=self.video_height)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path", default="./assets/apple.mp4", help="path to a video")
    parser.add_argument("--json_path", default=None,
                        help="path to a JSON file containing annotations")
    parser.add_argument("--grid_size", type=int,
                        default=10, help="Regular grid size")
    parser.add_argument("--grid_query_frame", type=int, default=0,
                        help="Compute dense and grid tracks starting from this frame")
    args = parser.parse_args()

    tracker_processor = CoTrackerProcessor(args.video_path, args.json_path)
    tracker_processor.process_video(args.grid_size, args.grid_query_frame)
