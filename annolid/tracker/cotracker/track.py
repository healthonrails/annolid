from __future__ import annotations
import argparse
import numpy as np
import torch

from annolid.tracker.point_tracking_processor import BasePointTrackingProcessor
from annolid.utils.logger import logger
from typing import Optional, Callable, Any

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


class CoTrackerProcessor(BasePointTrackingProcessor):
    """Thin wrapper around the official CoTracker model with Annolid hooks."""

    supports_online = True

    def __init__(
        self,
        video_path: str,
        json_path: Optional[str] = None,
        is_online: bool = True,
        should_stop: Optional[Callable[[], bool]] = None,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            video_path=video_path,
            json_path=json_path,
            is_online=is_online,
            should_stop=should_stop,
            model_name=model_name,
            **kwargs,
        )

    def load_model(self):
        model_name = "cotracker3_online" if self.is_online else "cotracker3_offline"
        cache_key = (model_name, str(self.device))
        if cache_key in _MODEL_CACHE:
            logger.debug(
                "Reusing cached CoTracker model '%s' on %s", model_name, self.device
            )
            return _MODEL_CACHE[cache_key].to(self.device)

        try:
            cotracker = torch.hub.load(
                "facebookresearch/co-tracker:release_cotracker3", model_name
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load CoTracker model '{model_name}'."
            ) from exc

        cotracker = cotracker.to(self.device).eval()
        _MODEL_CACHE[cache_key] = cotracker
        logger.info("Loaded CoTracker model '%s' on %s", model_name, self.device)
        return cotracker

    def _build_chunk_queries(
        self, chunk_start_frame: int, chunk_num_frames: int
    ) -> Optional[torch.Tensor]:
        """Map global query frame indices into the local chunk frame range."""
        if self.queries is None or chunk_num_frames <= 0:
            return None

        q = self.queries.clone()
        # CoTracker expects query timestamps in [0, T-1] for the provided chunk.
        q[:, 0] = torch.clamp(
            q[:, 0] - float(chunk_start_frame),
            min=0.0,
            max=float(chunk_num_frames - 1),
        )
        return q[None]

    def process_step(
        self,
        window_frames,
        is_first_step,
        grid_size,
        grid_query_frame,
        chunk_start_frame: int,
    ):
        model = self._ensure_model()
        window = window_frames[-model.step * 2 :]
        video_chunk = (
            torch.tensor(np.stack(window), device=self.device)
            .float()
            .permute(0, 3, 1, 2)[None]
        )

        kwargs = {
            "is_first_step": is_first_step,
            "grid_size": grid_size,
            "grid_query_frame": grid_query_frame,
        }
        queries = self._build_chunk_queries(
            chunk_start_frame=chunk_start_frame,
            chunk_num_frames=video_chunk.shape[1],
        )
        if queries is not None:
            kwargs["queries"] = queries

        return model(video_chunk, **kwargs)

    def _process_video_online(self, grid_size, grid_query_frame, need_visualize):
        """Process video using CoTracker online mode with incremental saving.

        Uses a hybrid approach:
        1. First window: Bidirectional processing to capture ALL early frames
        2. Remaining frames: Online mode with incremental saving

        This ensures complete coverage from start_frame onwards.
        """
        del need_visualize
        model = self._ensure_model()
        total_vid_frames = self.video_loader.total_frames()
        actual_end = self.end_frame if self.end_frame >= 0 else total_vid_frames - 1
        step_size = model.step
        min_window_size = step_size * 2

        logger.info(
            "CoTracker online: step=%d, min_window=%d, frames %d-%d",
            step_size,
            min_window_size,
            self.start_frame,
            actual_end,
        )

        # Phase 1: Process first window with bidirectional mode to get all early frames
        first_window_end = min(self.start_frame + min_window_size - 1, actual_end)

        pred_tracks, pred_visibility = self._process_initial_window_bidirectional(
            self.start_frame, first_window_end, grid_size, grid_query_frame
        )

        if pred_tracks is None:
            return self._stop_message(self.start_frame)

        # Save first window frames
        num_local_frames = int(pred_tracks.shape[1])
        self.extract_frame_points(
            pred_tracks,
            pred_visibility,
            chunk_start_frame=self.start_frame,
            local_frame_indices=range(num_local_frames),
        )
        last_saved_frame = self.start_frame + num_local_frames - 1
        logger.info(
            "CoTracker: saved initial window frames %d-%d",
            self.start_frame,
            last_saved_frame,
        )

        if self._should_stop() or last_saved_frame >= actual_end:
            return f"Completed. Saved frames {self.start_frame}-{last_saved_frame}"

        # Phase 2: Continue with online mode for remaining frames
        window_frames = []
        is_first_step = True  # First step for online continuation
        frame_idx = last_saved_frame

        # Preload the last min_window_size frames into the window buffer
        for i in range(
            max(self.start_frame, last_saved_frame - min_window_size + 1),
            last_saved_frame + 1,
        ):
            frame = self.video_loader.load_frame(i)
            if frame is not None:
                window_frames.append(frame)

        for i in range(last_saved_frame + 1, actual_end + 1):
            if self._should_stop():
                self._stop_triggered = True
                logger.info("CoTracker stop requested at frame %s", i)
                break

            frame = self.video_loader.load_frame(i)
            if frame is None:
                logger.warning("Failed to load frame %s, stopping", i)
                break

            frame_idx = i
            window_frames.append(frame)

            # Process every step_size frames
            if (
                len(window_frames) >= min_window_size
                and (i - last_saved_frame) % step_size == 0
            ):
                pred_tracks, pred_visibility = self.process_step(
                    window_frames,
                    is_first_step,
                    grid_size,
                    grid_query_frame,
                    chunk_start_frame=i - min_window_size + 1,
                )

                if pred_tracks is not None:
                    num_local_frames = int(pred_tracks.shape[1])
                    chunk_start_frame = max(self.start_frame, i - num_local_frames + 1)
                    local_start = max(0, (last_saved_frame + 1) - chunk_start_frame)

                    if local_start < num_local_frames:
                        self.extract_frame_points(
                            pred_tracks,
                            pred_visibility,
                            chunk_start_frame=chunk_start_frame,
                            local_frame_indices=range(local_start, num_local_frames),
                        )
                        last_saved_frame = chunk_start_frame + num_local_frames - 1

                is_first_step = False

            # Maintain sliding window
            if len(window_frames) > min_window_size:
                window_frames = window_frames[-min_window_size:]

        # Final flush for remaining frames
        if last_saved_frame < frame_idx and len(window_frames) > 0:
            pred_tracks, pred_visibility = self.process_step(
                window_frames,
                is_first_step,
                grid_size,
                grid_query_frame,
                chunk_start_frame=frame_idx
                - min(len(window_frames), min_window_size)
                + 1,
            )
            if pred_tracks is not None:
                num_local_frames = int(pred_tracks.shape[1])
                chunk_start_frame = max(
                    self.start_frame, frame_idx - num_local_frames + 1
                )
                local_start = max(0, (last_saved_frame + 1) - chunk_start_frame)
                if local_start < num_local_frames:
                    self.extract_frame_points(
                        pred_tracks,
                        pred_visibility,
                        chunk_start_frame=chunk_start_frame,
                        local_frame_indices=range(local_start, num_local_frames),
                    )
                    last_saved_frame = chunk_start_frame + num_local_frames - 1

        message = f"Completed. Saved frames {self.start_frame}-{last_saved_frame}"
        logger.info(message)
        return message

    def _process_initial_window_bidirectional(
        self, start_frame: int, end_frame: int, grid_size: int, grid_query_frame: int
    ):
        """Process initial window with bidirectional tracking.

        Uses offline bidirectional model to ensure ALL frames in the window
        get valid tracks, including the earliest frames.
        """
        video = self.video_loader.get_frames_between(start_frame, end_frame)
        video_tensor = (
            torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(self.device)
        )

        # Use the offline model for bidirectional processing
        try:
            offline_model = (
                torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
                .to(self.device)
                .eval()
            )
        except Exception:
            # Fall back to using online model with is_first_step=True
            logger.warning(
                "Could not load offline model, using online model for initial window"
            )
            model = self._ensure_model()
            return model(
                video_tensor,
                is_first_step=True,
                grid_size=grid_size,
                grid_query_frame=grid_query_frame,
                queries=self._build_chunk_queries(
                    chunk_start_frame=start_frame,
                    chunk_num_frames=video_tensor.shape[1],
                ),
            )

        kwargs = {"grid_size": grid_size, "grid_query_frame": grid_query_frame}
        queries = self._build_chunk_queries(
            chunk_start_frame=start_frame,
            chunk_num_frames=video_tensor.shape[1],
        )
        if queries is not None:
            kwargs["queries"] = queries

        with torch.no_grad():
            pred_tracks, pred_visibility = offline_model(video_tensor, **kwargs)

        return pred_tracks, pred_visibility

    def _process_video_bidirectional(
        self, start_frame=0, end_frame=60, grid_size=10, grid_query_frame=0
    ):
        if self._should_stop():
            self._stop_triggered = True
            return None, None, None

        logger.info(
            "Running bidirectional CoTracker: grid_size=%s, query_frame=%s, mask=%s",
            grid_size,
            grid_query_frame,
            self.mask_label,
        )

        video = self.video_loader.get_frames_between(start_frame, end_frame)
        video = (
            torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(self.device)
        )
        model = self._ensure_model()
        queries = self._build_chunk_queries(
            chunk_start_frame=start_frame,
            chunk_num_frames=video.shape[1],
        )
        pred_tracks, pred_visibility = model(
            video,
            grid_size=grid_size,
            queries=queries,
            backward_tracking=True,
            segm_mask=self.mask,
        )
        return pred_tracks, pred_visibility, video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path", default="./assets/apple.mp4", help="path to a video"
    )
    parser.add_argument(
        "--json_path", default=None, help="path to a JSON file containing annotations"
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument("--start_frame", type=int, default=0, help="Regular grid size")
    parser.add_argument(
        "--end_frame",
        type=int,
        default=60,
        help="Compute dense and grid tracks starting from this frame",
    )
    args = parser.parse_args()

    tracker_processor = CoTrackerProcessor(
        args.video_path, args.json_path, is_online=False
    )
    tracker_processor.process_video_frames(
        args.start_frame, args.end_frame, args.grid_size, args.grid_query_frame
    )
