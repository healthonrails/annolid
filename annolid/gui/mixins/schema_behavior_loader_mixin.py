from __future__ import annotations

from pathlib import Path
from typing import List

from annolid.annotation.pose_schema import (
    PoseSchema,
    find_pose_schema_in_dir,
)
from annolid.domain import (
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

    def _apply_loaded_pose_schema(
        self,
        schema: PoseSchema | None,
        schema_path: str | None = None,
        *,
        replace_order: bool = False,
    ) -> None:
        """Apply a resolved ``PoseSchema`` to window state, canvas, and sequencer."""
        self._pose_schema = schema
        self._pose_schema_path = str(schema_path) if schema_path else None
        try:
            canvas = getattr(self, "canvas", None)
            if canvas is not None and hasattr(canvas, "setPoseSchema"):
                canvas.setPoseSchema(self._pose_schema)
        except Exception:
            pass
        try:
            widget = getattr(self, "keypoint_sequence_widget", None)
            if widget is not None:
                widget.set_pose_schema(
                    self._pose_schema,
                    self._pose_schema_path,
                    replace_order=bool(
                        replace_order
                        and schema is not None
                        and bool(getattr(schema, "keypoints", None))
                    ),
                )
        except Exception:
            pass

    def _auto_load_pose_schema_from_dir(
        self,
        directory: str | Path | None,
        *,
        overwrite: bool = False,
        search_parent: bool = False,
    ) -> bool:
        """Auto-discover and load ``pose_schema.{json,yaml,yml}`` from ``directory``."""
        if not directory:
            return False
        if not overwrite and getattr(self, "_pose_schema", None) is not None:
            return False
        candidate = find_pose_schema_in_dir(directory, search_parent=search_parent)
        if candidate is None:
            return False
        try:
            loaded = PoseSchema.load(candidate)
        except Exception as exc:
            logger.debug("Skipping invalid pose schema %s: %s", candidate, exc)
            return False
        self._apply_loaded_pose_schema(
            loaded,
            str(candidate),
            replace_order=bool(overwrite),
        )
        return True

    def _configure_pose_schema_from_project(self) -> None:
        schema = self.project_schema
        resolved_schema: PoseSchema | None = None
        resolved_path: str | None = None

        if schema is not None:
            embedded = getattr(schema, "pose_schema", None)
            schema_path_value = getattr(schema, "pose_schema_path", None)
            if embedded and isinstance(embedded, dict):
                try:
                    resolved_schema = PoseSchema.from_dict(embedded)
                except Exception:
                    resolved_schema = None

            if schema_path_value:
                try:
                    p = Path(schema_path_value).expanduser()
                    if not p.is_absolute() and self.project_schema_path:
                        p = self.project_schema_path.parent / p
                    resolved_path = str(p)
                    if resolved_schema is None and p.is_file():
                        try:
                            resolved_schema = PoseSchema.load(p)
                        except Exception as exc:
                            logger.warning(
                                "Failed to load pose schema from %s: %s", p, exc
                            )
                except Exception:
                    resolved_path = str(schema_path_value)

        # Fallback: auto-discover pose_schema.{json,yaml,yml} near the active video/results folder.
        if resolved_schema is None:
            search_dirs: list[Path] = []
            for attr in ("video_results_folder", "annotation_dir"):
                folder = getattr(self, attr, None)
                if folder:
                    search_dirs.append(Path(folder))
            if getattr(self, "project_schema_path", None):
                search_dirs.append(Path(self.project_schema_path).parent)
            if getattr(self, "video_file", None):
                search_dirs.append(Path(self.video_file).parent)
            for folder in search_dirs:
                found = find_pose_schema_in_dir(folder)
                if found is not None:
                    try:
                        resolved_schema = PoseSchema.load(found)
                        resolved_path = str(found)
                        break
                    except Exception as exc:
                        logger.debug("Skipping invalid pose schema %s: %s", found, exc)

        self._apply_loaded_pose_schema(resolved_schema, resolved_path)

    def _load_behavior(self, behavior_csv_file: str) -> None:
        """Load behavior events from Annolid CSV, BORIS CSV/TSV, or .boris project."""
        from annolid.behavior.boris_io import import_boris_project
        from annolid.behavior.event_utils import ParsedEventRow, read_parsed_event_rows

        source_path = Path(behavior_csv_file)
        fps = self.fps if self.fps and self.fps > 0 else 29.97

        if source_path.suffix.lower() in (".h5", ".hdf", ".hdf5"):
            if not self._load_deeplabcut_table(behavior_csv_file):
                logger.debug(
                    "Skipped loading '%s' because it is not a supported DeepLabCut HDF5 export.",
                    source_path.name,
                )
            return

        if source_path.suffix.lower() == ".boris":
            try:
                imported = import_boris_project(source_path)
            except Exception as exc:
                logger.warning(
                    "Failed to import BORIS project '%s': %s",
                    source_path.name,
                    exc,
                )
                return
            self.project_schema = imported.schema
            self.behavior_controller.configure_from_schema(imported.schema)
            self._populate_behavior_controls_from_schema(imported.schema)
            self._sync_behavior_flags_from_schema(imported.schema)
            valid_rows: List[ParsedEventRow] = []
            if imported.observations:
                obs = imported.observations[0]
                if obs.fps and obs.fps > 0 and not (self.fps and self.fps > 0):
                    fps = float(obs.fps)
                valid_rows = [
                    r for r in obs.events if r.recording_time is not None and r.behavior
                ]
                apply_sync = getattr(self, "_apply_boris_observation_sync_assets", None)
                if callable(apply_sync):
                    apply_sync(obs, source_path.parent)
        else:
            try:
                parsed_rows = read_parsed_event_rows(source_path)
            except ValueError:
                if not self._load_deeplabcut_table(behavior_csv_file):
                    logger.debug(
                        "Skipped loading '%s' because it is neither a behavior log nor a DeepLabCut export.",
                        source_path.name,
                    )
                return

            valid_rows = []
            for row in parsed_rows:
                if row.recording_time is None:
                    logger.warning(
                        "Failed to convert timestamp '%s' for behavior '%s'.",
                        row.raw_recording_time,
                        row.behavior,
                    )
                    continue
                if not row.behavior:
                    continue
                valid_rows.append(row)

        def time_to_frame(time_value: float) -> int:
            return int(round(time_value * fps))

        self.behavior_controller.load_events_from_rows(
            valid_rows,
            time_to_frame=time_to_frame,
        )
        if getattr(self, "seekbar", None) is not None:
            self.behavior_controller.attach_slider(self.seekbar)
        fps_for_log = self.fps if self.fps and self.fps > 0 else fps
        if getattr(self, "behavior_log_widget", None) is not None:
            self.behavior_log_widget.set_events(
                list(self.behavior_controller.iter_events()),
                fps=fps_for_log,
            )
        current = dict(self.pinned_flags or {})
        for behavior in self.behavior_controller.behavior_names:
            current[str(behavior)] = False
        self.loadFlags(current)
        refresh_badges = getattr(self, "_refresh_ethogram_active_badges", None)
        if callable(refresh_badges):
            refresh_badges()
        refresh_overlay = getattr(self, "_refresh_behavior_overlay", None)
        if callable(refresh_overlay):
            refresh_overlay()

    def _clear_deeplabcut_table(self) -> None:
        self._df_deeplabcut = None
        self._df_deeplabcut_columns = None
        self._df_deeplabcut_scorer = None
        self._df_deeplabcut_bodyparts = None
        self._df_deeplabcut_animal_ids = None
        self._df_deeplabcut_multi_animal = False

    def _load_deeplabcut_table(self, behavior_csv_file: str) -> bool:
        """Load DeepLabCut / Lightning Pose tracking results (3- or 4-level CSV or H5)."""
        import pandas as pd

        from annolid.annotation.pose_interop.deeplabcut import parse_dlc_table

        try:
            table = parse_dlc_table(behavior_csv_file, for_training=False)
        except Exception as exc:
            logger.debug(
                "Skipping %s: not a DeepLabCut multi-index table (%s).",
                Path(behavior_csv_file).name,
                exc,
            )
            self._clear_deeplabcut_table()
            return False

        df_dlc = table.df
        index_numeric = pd.to_numeric(df_dlc.index, errors="coerce")
        if index_numeric.isna().any():
            logger.debug(
                "DeepLabCut table %s has non-numeric frame index; using positional indices.",
                Path(behavior_csv_file).name,
            )
            df_dlc = df_dlc.reset_index(drop=True)
        else:
            df_dlc.index = index_numeric.astype(int)

        self._df_deeplabcut = df_dlc
        self._df_deeplabcut_columns = df_dlc.columns
        self._df_deeplabcut_scorer = table.scorer
        self._df_deeplabcut_bodyparts = list(table.bodyparts)
        self._df_deeplabcut_animal_ids = (
            list(table.individuals) if table.multi_animal else [None]
        )
        self._df_deeplabcut_multi_animal = bool(table.multi_animal)
        if getattr(self, "_pose_schema", None) is None:
            source_dir = Path(behavior_csv_file).expanduser().parent
            if not self._auto_load_pose_schema_from_dir(
                source_dir, overwrite=False, search_parent=True
            ):
                try:
                    from annolid.annotation.pose_interop.deeplabcut import (
                        discover_dlc_config_near,
                    )

                    cfg = discover_dlc_config_near(behavior_csv_file)
                    keypoints = list(
                        (cfg.bodyparts if cfg and cfg.bodyparts else None)
                        or table.bodyparts
                        or ()
                    )
                    if keypoints:
                        edges = (
                            list(cfg.skeleton)
                            if (cfg and cfg.skeleton)
                            else PoseSchema.infer_edges(keypoints, preset="auto")
                        )
                        instances = (
                            list(table.individuals)
                            if table.multi_animal
                            else (
                                list(cfg.individuals) if cfg and cfg.individuals else []
                            )
                        )
                        dlc_schema = PoseSchema(
                            keypoints=keypoints,
                            edges=edges,
                            symmetry_pairs=PoseSchema.infer_symmetry_pairs(keypoints),
                            instances=instances,
                        )
                        self._apply_loaded_pose_schema(dlc_schema, None)
                except Exception as exc:
                    logger.debug(
                        "Unable to infer PoseSchema from DeepLabCut overlay %s: %s",
                        behavior_csv_file,
                        exc,
                    )
        return True
