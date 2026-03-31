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
        self._tracking_frame_slices: dict[int, tuple[int, int]] | None = None
        self._tracking_frame_indices: dict[int, tuple[int, ...]] | None = None

    @property
    def tracking_dataframe(self) -> pd.DataFrame | None:
        return self._tracking_df

    def tracking_rows_for_frame(self, frame_number: int) -> list[dict]:
        """Return tracking rows for a frame using the precomputed frame cache."""
        df = self._tracking_df
        if df is None or df.empty:
            return []

        try:
            frame_key = int(frame_number)
        except (TypeError, ValueError):
            return []

        frame_slices = self._tracking_frame_slices or {}
        frame_slice = frame_slices.get(frame_key)
        if frame_slice is not None:
            start, end = frame_slice
            try:
                return df.iloc[start:end].to_dict(orient="records")
            except Exception:
                # Fall back to the indexed path below if the CSV was reloaded or
                # the slice cache is stale.
                pass

        frame_indices = self._tracking_frame_indices or {}
        indices = frame_indices.get(frame_key)
        if not indices:
            return []

        try:
            frame_df = df.iloc[list(indices)]
        except Exception:
            # Best-effort fallback to the original DataFrame if the cached index
            # became stale or the CSV was reloaded unexpectedly.
            try:
                frame_df = df[df.frame_number == frame_key]
            except Exception:
                return []

        return frame_df.to_dict(orient="records")

    def _build_tracking_frame_slices(
        self, df: pd.DataFrame
    ) -> dict[int, tuple[int, int]]:
        if "frame_number" not in df.columns or df.empty:
            return {}

        try:
            frame_series = df["frame_number"]
            if not frame_series.is_monotonic_increasing:
                return {}
            frame_values = frame_series.to_numpy()
        except Exception:
            return {}

        if len(frame_values) == 0:
            return {}

        change_points = [0]
        try:
            change_points.extend(
                int(idx) + 1
                for idx in (frame_values[1:] != frame_values[:-1]).nonzero()[0]
            )
        except Exception:
            return {}
        change_points.append(len(frame_values))

        frame_slices: dict[int, tuple[int, int]] = {}
        for start, end in zip(change_points[:-1], change_points[1:]):
            try:
                frame_key = int(frame_values[start])
            except (TypeError, ValueError, IndexError):
                continue
            frame_slices[frame_key] = (int(start), int(end))
        return frame_slices

    def _build_tracking_frame_index(
        self, df: pd.DataFrame
    ) -> dict[int, tuple[int, ...]]:
        if "frame_number" not in df.columns or df.empty:
            return {}

        try:
            grouped = df.groupby("frame_number", sort=False).indices
        except Exception:
            return {}

        frame_indices: dict[int, tuple[int, ...]] = {}
        for frame_number, indices in grouped.items():
            try:
                frame_key = int(frame_number)
            except (TypeError, ValueError):
                continue
            try:
                frame_indices[frame_key] = tuple(int(idx) for idx in indices)
            except Exception:
                continue
        return frame_indices

    def load_tracking_results(
        self, cur_video_folder: Path, video_filename: str
    ) -> None:
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
                logger.info("Loading main tracking data from: %s", main_tracking_file)
                df = pd.read_csv(main_tracking_file)
                if "frame_number" not in df.columns and "Unnamed: 0" in df.columns:
                    df.rename(columns={"Unnamed: 0": "frame_number"}, inplace=True)
                if "frame_number" in df.columns:
                    self._tracking_df = df
                    self._tracking_frame_slices = self._build_tracking_frame_slices(df)
                    self._tracking_frame_indices = self._build_tracking_frame_index(df)
                    w._df = df
                else:
                    logger.warning(
                        "'%s' is missing the required 'frame_number' column.",
                        main_tracking_file,
                    )
                    self._tracking_frame_slices = None
                    self._tracking_frame_indices = None
            except Exception as exc:
                logger.error(
                    "Error loading main tracking file %s: %s", main_tracking_file, exc
                )
                self._tracking_frame_slices = None
                self._tracking_frame_indices = None

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
            if (
                isinstance(results_dir, Path)
                and results_dir.exists()
                and results_dir != cur_video_folder
            ):
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
        if not loaded_behavior:
            try:
                w.behavior_controller.load_events_from_store()
            except Exception as exc:
                logger.debug(
                    "Failed to load behavior events from annotation store: %s", exc
                )
            else:
                if w.behavior_controller.events_count:
                    w.behavior_controller.attach_slider(w.seekbar)
                    fps_for_log = w.fps if w.fps and w.fps > 0 else 29.97
                    w.behavior_log_widget.set_events(
                        list(w.behavior_controller.iter_events()),
                        fps=fps_for_log,
                    )
                    w.pinned_flags.update(
                        {
                            behavior: False
                            for behavior in w.behavior_controller.behavior_names
                        }
                    )
                    loaded_behavior = True

        if labels_file_path.is_file():
            logger.info("Loading labels data from: %s", labels_file_path)
            w._load_labels(labels_file_path)
