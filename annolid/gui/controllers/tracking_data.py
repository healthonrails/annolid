from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from annolid.utils.logger import logger

if TYPE_CHECKING:
    from annolid.gui.app import AnnolidWindow


class TrackingDataController:
    """Handle loading of tracking/behavior CSV data for the active video."""

    def __init__(self, window: "AnnolidWindow") -> None:
        self._window = window
        self._tracking_df: pd.DataFrame | None = None

    @property
    def tracking_dataframe(self) -> pd.DataFrame | None:
        return self._tracking_df

    def load_tracking_results(self, cur_video_folder: Path, video_filename: str) -> None:
        w = self._window

        w.behavior_controller.clear()
        w.behavior_log_widget.clear()
        w.pinned_flags = {}
        w._df = None
        self._tracking_df = None

        video_name = Path(video_filename).stem
        main_tracking_file = cur_video_folder / f"{video_name}_tracking.csv"
        timestamps_file = cur_video_folder / f"{video_name}_timestamps.csv"
        labels_file_path = cur_video_folder / f"{video_name}_labels.csv"

        if main_tracking_file.is_file():
            try:
                logger.info("Loading main tracking data from: %s",
                            main_tracking_file)
                df = pd.read_csv(main_tracking_file)
                if 'frame_number' not in df.columns and 'Unnamed: 0' in df.columns:
                    df.rename(
                        columns={'Unnamed: 0': 'frame_number'}, inplace=True)
                if 'frame_number' in df.columns:
                    self._tracking_df = df
                    w._df = df
                else:
                    logger.warning(
                        "'%s' is missing the required 'frame_number' column.",
                        main_tracking_file,
                    )
            except Exception as exc:
                logger.error(
                    "Error loading main tracking file %s: %s", main_tracking_file, exc)

        if timestamps_file.is_file():
            logger.info("Loading behavior data from: %s", timestamps_file)
            w._load_behavior(timestamps_file)

        if labels_file_path.is_file():
            logger.info("Loading labels data from: %s", labels_file_path)
            w._load_labels(labels_file_path)
