from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any, Callable, Optional

import numpy as np
import torch

from annolid.tracker.base import BasePointTracker
from annolid.utils.logger import logger


class BasePointTrackingProcessor(BasePointTracker):
    """Reusable processing workflow for point-tracking backends."""

    supports_online = False

    def __init__(
        self,
        video_path: str,
        json_path: Optional[str] = None,
        is_online: bool = False,
        should_stop: Optional[Callable[[], bool]] = None,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            video_path=video_path,
            json_path=json_path,
            should_stop=should_stop,
            model_name=model_name,
            **kwargs,
        )
        self.is_online = bool(is_online and self.supports_online)
        self.model = None
        self.queries = None

    def _ensure_model(self):
        if self.model is None:
            self.model = self.load_model()
        return self.model

    def process_video_frames(
        self,
        start_frame: int = 0,
        end_frame: int = -1,
        grid_size: int = 10,
        grid_query_frame: int = 0,
        need_visualize: bool = False,
        **kwargs: Any,
    ) -> str:
        if not os.path.isfile(self.video_path):
            raise ValueError("Video file does not exist")

        logger.info(
            "Processing %s from frame %s to %s",
            self.__class__.__name__,
            start_frame,
            end_frame,
        )
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.queries = self.load_queries()
        self._stop_triggered = False

        if self._should_stop():
            logger.info(
                "%s stop requested before processing started", self.__class__.__name__
            )
            return self._stop_message(start_frame)

        if self.is_online:
            return self._process_video_online(
                grid_size=grid_size,
                grid_query_frame=grid_query_frame,
                need_visualize=need_visualize,
            )

        pred_tracks, pred_visibility, video_source = self._process_video_bidirectional(
            start_frame=start_frame,
            end_frame=end_frame,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
        )
        if self._stop_triggered or pred_tracks is None:
            return self._stop_message(start_frame)
        return self._finalize_tracking(
            pred_tracks,
            pred_visibility,
            video_source,
            grid_query_frame,
            need_visualize,
        )

    def _process_video_online(
        self,
        grid_size: int,
        grid_query_frame: int,
        need_visualize: bool,
    ) -> str:
        raise RuntimeError(
            f"{self.__class__.__name__} does not implement online point tracking."
        )

    @abstractmethod
    def _process_video_bidirectional(
        self,
        start_frame: int = 0,
        end_frame: int = 60,
        grid_size: int = 10,
        grid_query_frame: int = 0,
    ):
        """Run batch/offline tracking and return tracks, visibility, and video source."""

    def _finalize_tracking(
        self,
        pred_tracks: torch.Tensor,
        pred_visibility: torch.Tensor,
        video_source: torch.Tensor | np.ndarray | list[np.ndarray],
        grid_query_frame: int,
        need_visualize: bool,
    ) -> str:
        if pred_tracks is None or pred_visibility is None:
            return self._stop_message(self.start_frame)

        message = self.extract_frame_points(
            pred_tracks,
            pred_visibility,
            chunk_start_frame=self.start_frame,
        )

        if need_visualize:
            self._visualize(
                tracks=pred_tracks,
                visibility=pred_visibility,
                video_source=video_source,
                query_frame=grid_query_frame,
            )
        return message

    def _visualize(
        self,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        video_source: torch.Tensor | np.ndarray | list[np.ndarray],
        query_frame: int,
    ) -> None:
        from annolid.tracker.cotracker.visualizer import Visualizer

        vis_video_name = f"{self.video_result_folder.name}_tracked"
        vis = Visualizer(
            save_dir=str(self.video_result_folder.parent),
            linewidth=6,
            mode="cool",
            tracks_leave_trace=-1,
        )
        if isinstance(video_source, list):
            video = torch.tensor(np.stack(video_source), device=self.device).permute(
                0, 3, 1, 2
            )[None]
            vis.visualize(
                video,
                tracks,
                visibility,
                query_frame=query_frame,
                filename=vis_video_name,
            )
        else:
            vis.visualize(
                video=video_source,
                tracks=tracks,
                visibility=visibility,
                filename=vis_video_name,
            )
