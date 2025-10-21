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

        def _discover_behavior_files(search_root: Path) -> list[Path]:
            candidates: list[Path] = []
            video_name_lower = video_name.lower()
            for candidate in search_root.glob(f"{video_name}*.csv"):
                name_lower = candidate.name.lower()
                if name_lower == f"{video_name_lower}_tracking.csv":
                    continue
                if name_lower == f"{video_name_lower}_labels.csv":
                    continue
                candidates.append(candidate)
            return sorted(candidates)

        behavior_candidates: list[Path] = []
        if timestamps_file.is_file():
            behavior_candidates.append(timestamps_file)
        else:
            behavior_candidates.extend(_discover_behavior_files(cur_video_folder))
            results_dir = getattr(w, "video_results_folder", None)
            if isinstance(results_dir, Path) and results_dir.exists() and results_dir != cur_video_folder:
                behavior_candidates.extend(_discover_behavior_files(results_dir))

        seen_paths: set[Path] = set()
        unique_candidates: list[Path] = []
        for candidate in behavior_candidates:
            if candidate not in seen_paths:
                unique_candidates.append(candidate)
                seen_paths.add(candidate)
        behavior_candidates = unique_candidates

        loaded_behavior = False
        for candidate in behavior_candidates:
            try:
                logger.info("Loading behavior data from: %s", candidate)
                w._load_behavior(str(candidate))
                loaded_behavior = True
                break
            except Exception as exc:
                logger.error("Failed to load behavior data from %s: %s", candidate, exc)

        if not loaded_behavior and behavior_candidates:
            logger.warning(
                "Behavior CSV files were discovered for '%s' but could not be loaded.",
                video_name,
            )
        elif not behavior_candidates:
            logger.debug(
                "No behavior CSV detected for '%s'. Looking for '%s' or similarly named files.",
                video_name,
                f"{video_name}_timestamps.csv",
            )

        if labels_file_path.is_file():
            logger.info("Loading labels data from: %s", labels_file_path)
            w._load_labels(labels_file_path)
