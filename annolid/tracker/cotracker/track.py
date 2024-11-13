import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np
import json
from pathlib import Path
from labelme.utils.shape import shape_to_mask
from annolid.annotation.keypoints import save_labels
from annolid.gui.shape import Shape
from annolid.tracker.cotracker.visualizer import Visualizer
from annolid.utils.logger import logger
from annolid.utils.files import get_frame_number_from_json
from annolid.utils.files import find_manual_labeled_json_files
from annolid.data.videos import CV2Video
from annolid.utils.shapes import sample_grid_in_polygon


"""
@article{karaev2023cotracker,
  title={CoTracker: It is Better to Track Together},
  author={Nikita Karaev and Ignacio Rocco and Benjamin Graham 
  and Natalia Neverova and Andrea Vedaldi and Christian Rupprecht},
  journal={arXiv:2307.07635},
  year={2023}
}

@inproceedings{karaev24cotracker3,
  title     = {CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos},
  author    = {Nikita Karaev and Iurii Makarov and Jianyuan Wang and Natalia Neverova and Andrea Vedaldi and Christian Rupprecht},
  booktitle = {Proc. {arXiv:2410.11831}},
  year      = {2024}
}

"""


class CoTrackerProcessor:
    def __init__(self, video_path, json_path=None, is_online=True):
        self.video_path = video_path
        self.video_loader = CV2Video(self.video_path)
        self.video_result_folder = Path(video_path).with_suffix('')
        self.total_num_frames = self.video_loader.total_frames()
        self.create_video_result_folder()
        self.point_labels = []
        self.mask = None
        self.mask_label = None
        first_frame = self.video_loader.get_first_frame()
        self.video_height, self.video_width, _ = first_frame.shape
        self.is_online = is_online
        self.start_frame = 0
        self.end_frame = 0
        self.predict_interval = 60
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.queries = None
        self.model = self.load_model()

    def create_video_result_folder(self):
        if not self.video_result_folder.exists():
            self.video_result_folder.mkdir(exist_ok=True)

    def load_model(self):
        model_name = "cotracker3_online" if self.is_online else "cotracker3_offline"
        cotracker = torch.hub.load(
            "facebookresearch/co-tracker:release_cotracker3", model_name).to(self.device)
        return cotracker

    def load_queries(self):
        """
        Load queries from JSON files and process shapes to extract queries.

        Returns:
            torch.Tensor: A tensor containing the selected queries.
        """
        json_files = find_manual_labeled_json_files(self.video_result_folder)
        queries = []

        for json_path in sorted(json_files):
            if not json_path:
                continue

            json_abs_path = self.video_result_folder / json_path
            with open(json_abs_path, 'r') as file:
                data = json.load(file)

            frame_number = get_frame_number_from_json(json_path)
            if not self.is_online and self.end_frame > 0 and frame_number <= self.end_frame:
                queries.extend(self._process_shapes(
                    data['shapes'], frame_number))
            else:
                queries.extend(self._process_shapes(
                    data['shapes'], frame_number))

        queries_tensor = torch.tensor(queries).float().to(self.device)
        return queries_tensor

    def _process_shapes(self, shapes, frame_number):
        """
        Process shapes to extract queries.

        Args:
            shapes (list): List of shapes in JSON format.
            frame_number (int): Frame number associated with the shapes.

        Returns:
            list: List of extracted queries.
        """
        processed_queries = []
        for shape in shapes:
            label = shape['label']
            shape_type = shape['shape_type']
            if shape_type == 'point':
                processed_queries.append(self._process_point(
                    shape['points'][0], frame_number, label))
            elif (shape_type == 'polygon' and "description" in shape
                  and shape["description"] is not None
                  and ('grid' in shape['description'] or
                       'point' in shape['description']
                       )
                  ):
                processed_queries.extend(self._process_polygon(
                    shape['points'], frame_number, label))

        return processed_queries

    def _process_point(self, point, frame_number, label):
        """
        Process a single point shape to extract query.

        Args:
            point (list): Coordinates of the point.
            frame_number (int): Frame number associated with the point.
            label (str): Label associated with the point.

        Returns:
            list: Extracted query.
        """
        self.point_labels.append(label)
        return [frame_number] + point

    def _process_polygon(self, points, frame_number, label):
        """
        Process a polygon shape to extract queries.

        Args:
            points (list): List of points defining the polygon.
            frame_number (int): Frame number associated with the polygon.
            label (str): Label associated with the polygon.

        Returns:
            list: List of extracted queries.
        """
        queries = []

        if self.mask is None:
            self.mask_label = label
            img_shape = (self.video_height, self.video_width)
            self.mask = shape_to_mask(
                img_shape, points, shape_type="polygon").astype(np.uint8)
            self.mask = torch.from_numpy(self.mask)[None, None].to(self.device)

        points_in_polygon = sample_grid_in_polygon(points)
        logger.info(f"Sampled {len(points_in_polygon)} points.")
        for point in points_in_polygon:
            self.point_labels.append(label)
            queries.append([frame_number] + list(point))

        return queries

    def process_step(self, window_frames,
                     is_first_step,
                     grid_size,
                     grid_query_frame):
        video_chunk = torch.tensor(np.stack(window_frames[-self.model.step * 2:]),
                                   device=self.device).float().permute(0, 3, 1, 2)[None]

        kwargs = {
            'is_first_step': is_first_step,
            'grid_size': grid_size,
            'grid_query_frame': grid_query_frame,
        }
        if self.queries is not None:
            kwargs['queries'] = self.queries[None]

        return self.model(video_chunk, **kwargs)

    def process_video(self,
                      start_frame=0,
                      end_frame=-1,
                      grid_size=10,
                      grid_query_frame=0,
                      need_visualize=False):
        if not os.path.isfile(self.video_path):
            raise ValueError("Video file does not exist")
        self.start_frame = start_frame
        self.end_frame = end_frame
        logger.info(f"Processing from frame {start_frame} to {end_frame}")
        self.queries = self.load_queries()

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
                if len(window_frames) > self.model.step * 2:
                    window_frames = window_frames[-self.model.step * 2:]

            pred_tracks, pred_visibility = self.process_step(
                window_frames[-(i % self.model.step) - self.model.step - 1:],
                is_first_step, grid_size, grid_query_frame
            )
        else:
            pred_tracks, pred_visibility, video = self._process_video_bidrection(
                start_frame, end_frame)

        logger.info("Tracks are computed")
        message = self.extract_frame_points(
            pred_tracks, pred_visibility, start_frame=start_frame)

        if need_visualize:
            vis_video_name = f'{self.video_result_folder.name}_tracked'
            vis = Visualizer(
                save_dir=str(self.video_result_folder.parent),
                linewidth=6,
                mode='cool',
                tracks_leave_trace=-1
            )
            if self.is_online:
                video = torch.tensor(
                    np.stack(window_frames), device=self.device).permute(0, 3, 1, 2)[None]
                vis.visualize(video, pred_tracks, pred_visibility,
                              query_frame=grid_query_frame, filename=vis_video_name)
            else:
                vis.visualize(video=video, tracks=pred_tracks,
                              visibility=pred_visibility, filename=vis_video_name)
        return message

    def _process_video_bidrection(self, start_frame=0,
                                  end_frame=60, grid_size=10,
                                  grid_query_frame=0):
        logger.info(
            f"grid_size: {grid_size}, grid_query_frame: {grid_query_frame}, mask label: {self.mask_label}")

        video = self.video_loader.get_frames_between(start_frame, end_frame)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[
            None].float().to(self.device)
        pred_tracks, pred_visibility = self.model(
            video,
            grid_size=grid_size,
            queries=self.queries[None],
            backward_tracking=True,
            segm_mask=self.mask,
        )
        return pred_tracks, pred_visibility, video

    def save_current_frame_tracked_points_to_json(self, frame_number, points):
        json_file_path = self.video_result_folder / \
            (self.video_result_folder.name + f"_{frame_number:0>{9}}.json")
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

    def extract_frame_points(self, tracks: torch.Tensor,
                             visibility: torch.Tensor = None,
                             query_frame: int = 0,
                             start_frame: int = 0):
        tracks = tracks[0].long().detach().cpu().numpy()
        for t in range(query_frame, tracks.shape[0]):
            _points = []
            for i in range(tracks.shape[1]):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                coord = np.array(coord)
                visible = True
                if visibility is not None:
                    visible = visibility[0, t, i].item()
                _points.append((coord, visible))
            self.save_current_frame_tracked_points_to_json(
                t + start_frame, _points)
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
    parser.add_argument("--start_frame", type=int,
                        default=0, help="Regular grid size")
    parser.add_argument("--end_frame", type=int, default=60,
                        help="Compute dense and grid tracks starting from this frame")
    args = parser.parse_args()

    tracker_processor = CoTrackerProcessor(
        args.video_path, args.json_path, is_online=False)
    tracker_processor.process_video(
        args.start_frame, args.end_frame, args.grid_size, args.grid_query_frame)
