from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from annolid.annotation.pose_schema import PoseSchema
from annolid.core.behavior.spec import (
    DEFAULT_SCHEMA_FILENAME,
    default_behavior_spec,
    load_behavior_spec,
    validate_behavior_spec,
)
from annolid.utils.logger import logger


class SchemaBehaviorLoaderMixin:
    """Project-schema and behavior/deeplabcut CSV loading helpers."""

    def _configure_project_schema_for_video(self, video_path: str) -> None:
        """Load optional project schema metadata located near the video."""
        schema, schema_path = load_behavior_spec(video_path=Path(video_path))
        if schema_path is not None:
            for warning in validate_behavior_spec(schema):
                logger.warning("Schema warning (%s): %s", schema_path.name, warning)
            self.project_schema_path = schema_path
        else:
            self.project_schema_path = (
                Path(video_path).with_suffix("") / DEFAULT_SCHEMA_FILENAME
            )
            logger.debug(
                "No project schema found near %s; using default configuration.",
                video_path,
            )
            schema = default_behavior_spec()
        self.project_schema = schema
        self.behavior_controller.configure_from_schema(schema)
        self._populate_behavior_controls_from_schema(schema)
        self._sync_behavior_flags_from_schema(schema)
        self._update_modifier_controls_for_behavior(self.event_type)
        self._configure_pose_schema_from_project()

    def _configure_pose_schema_from_project(self) -> None:
        schema = self.project_schema
        self._pose_schema = None
        self._pose_schema_path = None
        if schema is None:
            return

        embedded = getattr(schema, "pose_schema", None)
        schema_path_value = getattr(schema, "pose_schema_path", None)
        if embedded and isinstance(embedded, dict):
            try:
                self._pose_schema = PoseSchema.from_dict(embedded)
            except Exception:
                self._pose_schema = None

        if schema_path_value:
            try:
                p = Path(schema_path_value)
                if not p.is_absolute() and self.project_schema_path:
                    p = self.project_schema_path.parent / p
                self._pose_schema_path = str(p)
            except Exception:
                self._pose_schema_path = str(schema_path_value)

        try:
            self.canvas.setPoseSchema(self._pose_schema)
        except Exception:
            pass

    def _load_behavior(self, behavior_csv_file: str) -> None:
        """Load behavior events from CSV and populate the slider timeline."""
        df_behaviors = pd.read_csv(behavior_csv_file)
        required_columns = {"Recording time", "Event", "Behavior"}

        if not required_columns.issubset(df_behaviors.columns):
            del df_behaviors
            if not self._load_deeplabcut_table(behavior_csv_file):
                logger.debug(
                    "Skipped loading '%s' because it is neither a behavior log nor a DeepLabCut export.",
                    Path(behavior_csv_file).name,
                )
            return

        rows: List[Tuple[float, float, Optional[str], str, str]] = []

        for _, row in df_behaviors.iterrows():
            raw_timestamp = row.get("Recording time")
            event_label = str(row.get("Event"))
            behavior = str(row.get("Behavior"))
            raw_subject = row.get("Subject")
            raw_trial_time = row.get("Trial time")

            try:
                timestamp_value = float(raw_timestamp)
            except (TypeError, ValueError):
                logger.warning(
                    "Failed to convert timestamp '%s' for behavior '%s'.",
                    raw_timestamp,
                    behavior,
                )
                continue

            trial_time_value: Optional[float]
            try:
                trial_time_value = (
                    float(raw_trial_time)
                    if raw_trial_time is not None and pd.notna(raw_trial_time)
                    else None
                )
            except (TypeError, ValueError):
                trial_time_value = None

            subject_value = None
            if raw_subject is not None and pd.notna(raw_subject):
                subject_value = str(raw_subject)

            rows.append(
                (
                    trial_time_value,
                    timestamp_value,
                    subject_value,
                    behavior,
                    event_label,
                )
            )

        fps = self.fps if self.fps and self.fps > 0 else 29.97

        def time_to_frame(time_value: float) -> int:
            return int(round(time_value * fps))

        self.behavior_controller.load_events_from_rows(
            rows,
            time_to_frame=time_to_frame,
        )
        self.behavior_controller.attach_slider(self.seekbar)
        fps_for_log = self.fps if self.fps and self.fps > 0 else 29.97
        self.behavior_log_widget.set_events(
            list(self.behavior_controller.iter_events()),
            fps=fps_for_log,
        )
        self.pinned_flags.update(
            {behavior: False for behavior in self.behavior_controller.behavior_names}
        )

    def _load_deeplabcut_table(self, behavior_csv_file: str) -> bool:
        """Load DeepLabCut tracking results stored as a multi-index CSV."""
        try:
            df_dlc = pd.read_csv(
                behavior_csv_file,
                header=[0, 1, 2],
                index_col=0,
            )
        except (ValueError, pd.errors.ParserError) as exc:
            logger.debug(
                "Skipping %s: not a DeepLabCut multi-index CSV (%s).",
                Path(behavior_csv_file).name,
                exc,
            )
            self._df_deeplabcut = None
            self._df_deeplabcut_columns = None
            self._df_deeplabcut_scorer = None
            self._df_deeplabcut_bodyparts = None
            self._df_deeplabcut_animal_ids = None
            self._df_deeplabcut_multi_animal = False
            return False

        nlevels = df_dlc.columns.nlevels
        if nlevels == 4:
            expected_names = ["scorer", "animal", "bodyparts", "coords"]
            self._df_deeplabcut_multi_animal = True
        elif nlevels == 3:
            expected_names = ["scorer", "bodyparts", "coords"]
            self._df_deeplabcut_multi_animal = False
        else:
            logger.debug(
                "Skipping %s: expected 3 or 4 column levels, found %s.",
                Path(behavior_csv_file).name,
                nlevels,
            )
            self._df_deeplabcut = None
            self._df_deeplabcut_columns = None
            self._df_deeplabcut_scorer = None
            self._df_deeplabcut_bodyparts = None
            self._df_deeplabcut_animal_ids = None
            self._df_deeplabcut_multi_animal = False
            return False

        df_dlc.columns = df_dlc.columns.set_names(expected_names)

        index_numeric = pd.to_numeric(df_dlc.index, errors="coerce")
        if index_numeric.isna().any():
            logger.debug(
                "DeepLabCut table %s has non-numeric frame index; using positional indices.",
                Path(behavior_csv_file).name,
            )
            df_dlc.reset_index(drop=True, inplace=True)
        else:
            df_dlc.index = index_numeric.astype(int)

        self._df_deeplabcut = df_dlc
        self._df_deeplabcut_columns = df_dlc.columns
        self._df_deeplabcut_scorer = None
        self._df_deeplabcut_bodyparts = None
        self._df_deeplabcut_animal_ids = None
        return True
