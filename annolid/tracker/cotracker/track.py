import argparse
import json
import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from labelme.utils.shape import shape_to_mask

from annolid.annotation.keypoints import save_labels
from annolid.data.videos import CV2Video
from annolid.gui.shape import Shape
from annolid.tracker.cotracker.visualizer import Visualizer
from annolid.utils.files import (find_manual_labeled_json_files,
                                 get_frame_number_from_json)
from annolid.utils.logger import logger
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


_MODEL_CACHE: dict[tuple[str, str], torch.nn.Module] = {}


class CoTrackerProcessor:
    """Thin wrapper around the official CoTracker model with Annolid hooks."""

    def __init__(self, video_path, json_path=None, is_online=True, should_stop=None):
        self.video_path = video_path
        self.video_loader = CV2Video(self.video_path)
        self.video_result_folder = Path(video_path).with_suffix('')
        self.total_num_frames = self.video_loader.total_frames()
        self.create_video_result_folder()

        first_frame = self.video_loader.get_first_frame()
        if first_frame is None:
            raise RuntimeError(
                "Video contains no frames for CoTracker to process.")
        self.video_height, self.video_width, _ = first_frame.shape

        self.is_online = is_online
        self.start_frame = 0
        self.end_frame = 0
        self.predict_interval = 60
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.manual_json_path = Path(json_path) if json_path else None

        self.point_labels: list[str] = []
        self.mask = None
        self.mask_label = None
        self.queries = None
        self.model = None
        self._should_stop = should_stop or (lambda: False)
        self._stop_triggered = False

    def create_video_result_folder(self):
        self.video_result_folder.mkdir(exist_ok=True)

    def _ensure_model(self):
        if self.model is None:
            self.model = self.load_model()
        return self.model

    def load_model(self):
        model_name = "cotracker3_online" if self.is_online else "cotracker3_offline"
        cache_key = (model_name, str(self.device))
        if cache_key in _MODEL_CACHE:
            logger.debug("Reusing cached CoTracker model '%s' on %s",
                         model_name, self.device)
            return _MODEL_CACHE[cache_key].to(self.device)

        try:
            cotracker = torch.hub.load(
                "facebookresearch/co-tracker:release_cotracker3", model_name)
        except Exception as exc:  # pragma: no cover - hub/network failure
            raise RuntimeError(
                f"Failed to load CoTracker model '{model_name}'.") from exc

        cotracker = cotracker.to(self.device).eval()
        _MODEL_CACHE[cache_key] = cotracker
        logger.info("Loaded CoTracker model '%s' on %s",
                    model_name, self.device)
        return cotracker

    def load_queries(self):
        """Gather point prompts from labeled JSON files."""
        self.point_labels = []
        self.mask = None
        self.mask_label = None

        discovered = []
        if self.manual_json_path and self.manual_json_path.exists():
            discovered.append(self.manual_json_path)

        for filename in find_manual_labeled_json_files(self.video_result_folder):
            candidate = self.video_result_folder / filename
            if candidate.exists():
                discovered.append(candidate)

        unique_files = []
        seen = set()
        for path in sorted(discovered):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_files.append(path)

        if not unique_files:
            raise RuntimeError(
                "CoTracker requires at least one manually labeled frame before tracking.")

        queries = []
        for json_path in unique_files:
            with open(json_path, "r") as file:
                data = json.load(file)

            frame_number = get_frame_number_from_json(json_path.name)
            queries.extend(self._process_shapes(
                data.get('shapes', []), frame_number))

        if not queries:
            raise RuntimeError(
                "CoTracker could not find point prompts in the supplied labeled frames.")

        return torch.as_tensor(queries, dtype=torch.float32, device=self.device)

    def _process_shapes(self, shapes, frame_number):
        processed_queries = []
        for shape in shapes:
            label = shape.get('label')
            shape_type = shape.get('shape_type')
            if shape_type == 'point' and shape.get('points'):
                processed_queries.append(self._process_point(
                    shape['points'][0], frame_number, label))
            elif (shape_type == 'polygon'
                  and shape.get("description")
                  and any(key in shape["description"] for key in ("grid", "point"))):
                processed_queries.extend(self._process_polygon(
                    shape.get('points', []), frame_number, label))
        return processed_queries

    def _process_point(self, point, frame_number, label):
        self.point_labels.append(label)
        return [frame_number] + list(point)

    def _process_polygon(self, points, frame_number, label):
        queries = []
        if not points:
            return queries
        if self.mask is None:
            self.mask_label = label
            img_shape = (self.video_height, self.video_width)
            self.mask = shape_to_mask(
                img_shape, points, shape_type="polygon").astype(np.uint8)
            self.mask = torch.from_numpy(self.mask)[None, None].to(self.device)

        points_in_polygon = sample_grid_in_polygon(points)
        logger.info("Sampled %d points for polygon '%s'.",
                    len(points_in_polygon), label)
        for point in points_in_polygon:
            self.point_labels.append(label)
            queries.append([frame_number] + list(point))
        return queries

    def process_step(self, window_frames, is_first_step, grid_size, grid_query_frame):
        model = self._ensure_model()
        window = window_frames[-model.step * 2:]
        video_chunk = torch.tensor(np.stack(window),
                                   device=self.device).float().permute(0, 3, 1, 2)[None]

        kwargs = {
            'is_first_step': is_first_step,
            'grid_size': grid_size,
            'grid_query_frame': grid_query_frame,
        }
        if self.queries is not None:
            kwargs['queries'] = self.queries[None]

        return model(video_chunk, **kwargs)

    def process_video(self,
                      start_frame=0,
                      end_frame=-1,
                      grid_size=10,
                      grid_query_frame=0,
                      need_visualize=False):
        if not os.path.isfile(self.video_path):
            raise ValueError("Video file does not exist")

        logger.info("Processing CoTracker from frame %s to %s",
                    start_frame, end_frame)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.queries = self.load_queries()
        model = self._ensure_model()
        self._stop_triggered = False

        if self._should_stop():
            logger.info("CoTracker stop requested before processing started")
            return self._stop_message(start_frame)

        if self.is_online:
            return self._process_video_online(model, grid_size, grid_query_frame, need_visualize)

        pred_tracks, pred_visibility, video = self._process_video_bidrection(
            start_frame, end_frame, grid_size, grid_query_frame)
        if self._stop_triggered or pred_tracks is None:
            return self._stop_message(start_frame)
        return self._finalize_tracking(pred_tracks, pred_visibility, video,
                                       grid_query_frame, need_visualize)

    def _process_video_online(self, model, grid_size, grid_query_frame, need_visualize):
        window_frames = []
        is_first_step = True
        frame_idx = -1
        last_saved_frame = self.start_frame - 1

        for i, frame in enumerate(iio.imiter(self.video_path, plugin="FFMPEG")):
            frame_idx = i
            if self._should_stop():
                self._stop_triggered = True
                logger.info("CoTracker stop requested at frame %s", i)
                break
            if self.end_frame >= 0 and i > self.end_frame:
                break
            if self.video_height is None or self.video_width is None:
                self.video_height, self.video_width, _ = frame.shape
            if i % model.step == 0 and i != 0:
                pred_tracks, pred_visibility = self.process_step(
                    window_frames, is_first_step, grid_size, grid_query_frame)
                if pred_tracks is not None:
                    if i % 100 == 0:
                        logger.info("Tracking frame %d: tracks %s, visibility %s",
                                    i, pred_tracks.shape, pred_visibility.shape)
                    num_local_frames = int(pred_tracks.shape[1])
                    chunk_start_frame = max(0, i - num_local_frames)
                    local_start = max(
                        0, (last_saved_frame + 1) - chunk_start_frame)
                    self.extract_frame_points(
                        pred_tracks,
                        pred_visibility,
                        chunk_start_frame=chunk_start_frame,
                        local_frame_indices=range(
                            local_start, num_local_frames),
                    )
                    last_saved_frame = max(
                        last_saved_frame, chunk_start_frame + num_local_frames - 1
                    )
                is_first_step = False
            window_frames.append(frame)
            if len(window_frames) > model.step * 2:
                window_frames = window_frames[-model.step * 2:]

        if self._stop_triggered:
            return self._stop_message(max(frame_idx, self.start_frame))

        if frame_idx < 0:
            logger.warning("CoTracker online processing ended without frames")
            return self._stop_message(self.start_frame)

        pred_tracks, pred_visibility = self.process_step(
            window_frames[-(frame_idx % model.step) - model.step - 1:],
            is_first_step,
            grid_size,
            grid_query_frame,
        )
        if pred_tracks is not None:
            num_local_frames = int(pred_tracks.shape[1])
            chunk_end_exclusive = frame_idx + 1
            chunk_start_frame = max(0, chunk_end_exclusive - num_local_frames)
            local_start = max(0, (last_saved_frame + 1) - chunk_start_frame)
            message = self.extract_frame_points(
                pred_tracks,
                pred_visibility,
                chunk_start_frame=chunk_start_frame,
                local_frame_indices=range(local_start, num_local_frames),
            )
        else:
            message = self._stop_message(max(frame_idx, self.start_frame))

        if need_visualize and pred_tracks is not None:
            vis_video_name = f'{self.video_result_folder.name}_tracked'
            vis = Visualizer(
                save_dir=str(self.video_result_folder.parent),
                linewidth=6,
                mode='cool',
                tracks_leave_trace=-1,
            )
            frames_for_vis = window_frames[-num_local_frames:] if window_frames else []
            if frames_for_vis:
                video = torch.tensor(
                    np.stack(frames_for_vis), device=self.device).permute(0, 3, 1, 2)[None]
                vis.visualize(
                    video,
                    pred_tracks,
                    pred_visibility,
                    query_frame=grid_query_frame,
                    filename=vis_video_name,
                )
        return message

    def _finalize_tracking(self, pred_tracks, pred_visibility, video_source,
                           grid_query_frame, need_visualize, model=None):
        if pred_tracks is None or pred_visibility is None:
            return self._stop_message(self.start_frame)
        logger.info("CoTracker tracks computed")
        message = self.extract_frame_points(
            pred_tracks,
            pred_visibility,
            chunk_start_frame=self.start_frame,
        )

        if need_visualize:
            model = model or self._ensure_model()
            vis_video_name = f'{self.video_result_folder.name}_tracked'
            vis = Visualizer(
                save_dir=str(self.video_result_folder.parent),
                linewidth=6,
                mode='cool',
                tracks_leave_trace=-1,
            )
            if isinstance(video_source, list):
                video = torch.tensor(
                    np.stack(video_source), device=self.device).permute(0, 3, 1, 2)[None]
                vis.visualize(video, pred_tracks, pred_visibility,
                              query_frame=grid_query_frame, filename=vis_video_name)
            else:
                vis.visualize(video=video_source, tracks=pred_tracks,
                              visibility=pred_visibility, filename=vis_video_name)
        return message

    def _process_video_bidrection(self, start_frame=0,
                                  end_frame=60, grid_size=10,
                                  grid_query_frame=0):
        if self._should_stop():
            self._stop_triggered = True
            return None, None, None

        logger.info("Running bidirectional CoTracker: grid_size=%s, query_frame=%s, mask=%s",
                    grid_size, grid_query_frame, self.mask_label)

        video = self.video_loader.get_frames_between(start_frame, end_frame)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[
            None].float().to(self.device)
        model = self._ensure_model()
        pred_tracks, pred_visibility = model(
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
                    width=self.video_width, height=self.video_height,
                    persist_json=False)
        if frame_number % 100 == 0:
            logger.info("Saved %s", json_file_path)

    def extract_frame_points(
        self,
        tracks: torch.Tensor,
        visibility: torch.Tensor | None = None,
        *,
        chunk_start_frame: int = 0,
        local_frame_indices=None,
    ):
        """Persist tracked points as per-frame JSON files.

        Notes:
        - `tracks` is indexed by *local* time (0..S-1). `chunk_start_frame` maps
          local indices to *global* video frame numbers.
        - This mapping is critical for online CoTracker inference where each
          inference step returns a sliding window of frames.
        """
        tracks_np = tracks[0].long().detach().cpu().numpy()  # S, N, 2
        num_local_frames = tracks_np.shape[0]

        if local_frame_indices is None:
            local_frame_indices = range(num_local_frames)

        last_saved_frame = None
        for local_t in local_frame_indices:
            if local_t < 0 or local_t >= num_local_frames:
                continue
            global_frame = chunk_start_frame + int(local_t)

            if global_frame < self.start_frame:
                continue
            if self.end_frame >= 0 and global_frame > self.end_frame:
                continue

            points = []
            for point_idx in range(tracks_np.shape[1]):
                coord = np.array((tracks_np[local_t, point_idx, 0],
                                  tracks_np[local_t, point_idx, 1]))
                visible = True
                if visibility is not None:
                    visible = visibility[0, local_t, point_idx].item()
                points.append((coord, visible))

            self.save_current_frame_tracked_points_to_json(
                global_frame, points)
            last_saved_frame = global_frame

        if last_saved_frame is None:
            message = f"Saved all json file #{max(self.start_frame, 0)}"
        else:
            message = f"Saved all json file #{last_saved_frame}"
        logger.info(message)
        return message

    def _stop_message(self, frame_number):
        return f"CoTracker stopped by user#{frame_number}"


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
