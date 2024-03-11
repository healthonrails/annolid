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
from annolid.utils.logger import logger
from annolid.tracker.cotracker.visualizer import read_video_from_path

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
    def __init__(self, video_path, json_path=None, is_online=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model(is_online=is_online)
        self.video_path = video_path
        self.video_result_folder = Path(self.video_path).with_suffix('')
        if not self.video_result_folder.exists():
            self.video_result_folder.mkdir(exist_ok=True)
        self.point_labels = []
        self.queries = self.load_queries(json_path)
        self.video_height = None
        self.video_width = None
        self.is_online = is_online

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

    def load_model(self, is_online=True):
        if is_online:
            return torch.hub.load("facebookresearch/co-tracker",
                                  "cotracker2_online").to(self.device)
        else:
            return torch.hub.load("facebookresearch/co-tracker",
                                  "cotracker2").to(self.device)

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

    def process_video(self,
                      grid_size=10,
                      grid_query_frame=0,
                      need_visualize=True):
        if not os.path.isfile(self.video_path):
            raise ValueError("Video file does not exist")
        if self.is_online:
            window_frames = []
            is_first_step = True

            for i, frame in enumerate(iio.imiter(self.video_path, plugin="FFMPEG")):
                if self.video_height is None or self.video_width is None:
                    self.video_height, self.video_width, _ = frame.shape
                if i % self.model.step == 0 and i != 0:
                    pred_tracks, pred_visibility = self.process_step(
                        window_frames, is_first_step, grid_size, grid_query_frame)
                    if pred_tracks is not None:
                        logger.info(
                            f"Tracking frame {i}, {pred_tracks.shape}, {pred_visibility.shape}")
                    is_first_step = False
                window_frames.append(frame)

            pred_tracks, pred_visibility = self.process_step(
                window_frames[-(i % self.model.step) - self.model.step - 1:],
                is_first_step, grid_size, grid_query_frame)
        else:
            pred_tracks, pred_visibility, video = self._process_video_bidrection()

        logger.info("Tracks are computed")
        message = self.extract_frame_points(
            pred_tracks, pred_visibility, query_frame=0)

        if need_visualize:
            vis_video_name = f'{self.video_result_folder.name}_tracked'
            vis = Visualizer(
                save_dir=str(self.video_result_folder.parent),
                linewidth=6,
                mode='cool',
                tracks_leave_trace=-1
            )
            if self.is_online:
                video = torch.tensor(np.stack(window_frames),
                                     device=self.device).permute(0, 3, 1, 2)[None]
                vis.visualize(video, pred_tracks, pred_visibility,
                              query_frame=grid_query_frame,
                              filename=vis_video_name
                              )

            else:
                vis.visualize(
                    video=video,
                    tracks=pred_tracks,
                    visibility=pred_visibility,
                    filename=vis_video_name
                )
        return message

    def _process_video_bidrection(self,
                                  grid_size=10,
                                  grid_query_frame=0):
        logger.info(
            f"grid_size: {grid_size}, grid_query_frame: {grid_query_frame}")

        video = read_video_from_path(self.video_path)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        pred_tracks, pred_visibility = self.model(
            video, queries=self.queries[None],
            backward_tracking=True)
        return pred_tracks, pred_visibility, video

    def save_current_frame_tracked_points_to_json(self, frame_number, points):
        json_file_path = self.video_result_folder / \
            (self.video_result_folder.name +
             f"_{frame_number:0>{9}}.json")
        label_list = []
        for label, _point in zip(self.point_labels, points):
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
        if frame_number % 100 == 0:
            logger.info(f"Saved {json_file_path}")

    def extract_frame_points(
        self,
        tracks: torch.Tensor,  # (B,T,N,2)
        visibility: torch.Tensor = None,  # (B, T, N, 1) bool
        query_frame: int = 0,
    ):
        # Prepare the dictionary to hold frame points and their visibility
        # frame_point_dict = defaultdict(list)
        tracks = tracks[0].long().detach().cpu().numpy()  # S, N, 2

        # Iterate over each frame
        for t in range(query_frame, tracks.shape[0]):
            _points = []
            for i in range(tracks.shape[1]):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                coord = np.array(coord)
                visible = True
                if visibility is not None:
                    visible = visibility[0, t, i].item()
                _points.append((coord, visible))
            self.save_current_frame_tracked_points_to_json(t, _points)
        message = f"Saved all json file #{tracks.shape[0]}"
        logger.info(message)
        return message


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

    tracker_processor = CoTrackerProcessor(
        args.video_path, args.json_path)
    tracker_processor.process_video(args.grid_size, args.grid_query_frame)
