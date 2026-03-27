import pandas as pd
import json
import itertools
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from annolid.data.videos import CV2Video
from annolid.utils.logger import logger
from annolid.postprocessing.zone_analysis_engine import GenericZoneEngine
from annolid.postprocessing.social_zone_metrics import (
    build_anchor_dataframe,
    compute_pairwise_centroid_summary,
    load_pose_keypoint_observations,
)
from annolid.postprocessing.zone_assay_profiles import (
    ZoneAssayProfile,
    filter_zone_specs_by_assay_profile,
    resolve_assay_profile,
)
from annolid.postprocessing.zone_schema import (
    ZoneShapeSpec,
    load_zone_shapes,
    zone_shape_bounds,
    zone_shape_distance_to_point,
)


class TrackingResultsAnalyzer:
    """
    A class to analyze tracking results
      and visualize time spent in zones for instances.

    Attributes:
        video_path (str): The path of the video file.
        zone_file (str): The path to the JSON file containing zone information.
        tracking_csv (str): The path to the tracking CSV file.
        tracked_csv (str): The path to the tracked CSV file.
        tracking_df (DataFrame): DataFrame containing tracking data.
        tracked_df (DataFrame): DataFrame containing tracked data.
        merged_df (DataFrame): DataFrame containing merged tracking and tracked data.
        distances_df (DataFrame): DataFrame containing distances between instances.
        zone_data (dict): Dictionary containing zone information loaded from the zone JSON file.
    """

    def __init__(self, video_path, zone_file=None, fps=None, assay_profile=None):
        """
        Initialize the TrackingResultsAnalyzer.

        Args:
            video_name (str): The name of the video.
            zone_file (str): The path to the JSON file containing zone information.
        """
        self.video_path = Path(video_path)
        self.tracking_csv = (
            self.video_path.parent / f"{self.video_path.stem}_tracking.csv"
        )
        self.tracked_csv = (
            self.video_path.parent / f"{self.video_path.stem}_tracked.csv"
        )
        self.zone_file = Path(zone_file) if zone_file else None
        self.fps = fps
        self.assay_profile: ZoneAssayProfile = resolve_assay_profile(assay_profile)
        if fps is None:
            self.fps = CV2Video(self.video_path).get_fps()
        self.zone_specs: list[ZoneShapeSpec] = []
        self.zone_shapes = None
        self.zone_time_dict = None
        self.zone_engine: GenericZoneEngine | None = None
        self.load_zone_json()

    def read_csv_files(self):
        """Read tracking and tracked CSV files into DataFrames."""
        self.tracking_df = pd.read_csv(self.tracking_csv)
        self.tracked_df = pd.read_csv(self.tracked_csv)

    def merge_and_calculate_distance(self):
        """Merge tracking and tracked dataframes based on
        frame number and instance name, and calculate distances."""
        self.read_csv_files()

        # Merge DataFrames based on frame number and instance name
        self.merged_df = pd.merge(
            self.tracking_df,
            self.tracked_df,
            on=["frame_number", "instance_name"],
            suffixes=("_tracking", "_tracked"),
        )

        # Calculate distance between different instances in the same frame
        distances = []
        for frame_number, frame_group in self.merged_df.groupby("frame_number"):
            instances_in_frame = frame_group["instance_name"].unique()
            instance_combinations = itertools.combinations(instances_in_frame, 2)
            for instance_combination in instance_combinations:
                instance1 = instance_combination[0]
                instance2 = instance_combination[1]
                instance1_data = frame_group[frame_group["instance_name"] == instance1]
                instance2_data = frame_group[frame_group["instance_name"] == instance2]
                for _, row1 in instance1_data.iterrows():
                    for _, row2 in instance2_data.iterrows():
                        distance = self.calculate_distance(
                            row1["cx_tracking"],
                            row1["cy_tracking"],
                            row2["cx_tracked"],
                            row2["cy_tracked"],
                        )
                        distances.append(
                            {
                                "frame_number": frame_number,
                                "instance_name_1": instance1,
                                "instance_name_2": instance2,
                                "distance": distance,
                            }
                        )

        self.distances_df = pd.DataFrame(distances)

    def calculate_distance(self, x1, y1, x2, y2):
        """
        Calculate the Euclidean distance between two points.

        Args:
            x1 (float): X-coordinate of the first point.
            y1 (float): Y-coordinate of the first point.
            x2 (float): X-coordinate of the second point.
            y2 (float): Y-coordinate of the second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def load_zone_json(self):
        """Load zone information from the JSON file."""
        zone_path = None

        potential_roots = [
            self.video_path.parent,
            self.video_path.parent / self.video_path.stem,
        ]

        if self.zone_file:
            zone_candidates = [self.zone_file]
            if not self.zone_file.is_absolute():
                for root in potential_roots:
                    zone_candidates.append(root / self.zone_file)
                    zone_candidates.append(root / self.zone_file.name)
            for candidate in zone_candidates:
                if candidate.is_file():
                    zone_path = candidate
                    break

        if zone_path is None:
            for root in potential_roots:
                if not root.is_dir():
                    continue
                json_candidates = sorted(root.glob("*.json"))
                for candidate in json_candidates:
                    try:
                        data = json.loads(candidate.read_text())
                    except Exception:
                        continue
                    if load_zone_shapes(data):
                        zone_path = candidate
                        break
                if zone_path is not None:
                    break

        if zone_path is None:
            results_dir = self.video_path.parent / self.video_path.stem
            legacy_candidates = sorted(
                results_dir.glob("*.json") if results_dir.is_dir() else []
            )
            for candidate in legacy_candidates:
                try:
                    data = json.loads(candidate.read_text())
                except Exception:
                    continue
                if load_zone_shapes(data):
                    zone_path = candidate
                    break

        if zone_path is None:
            logger.warning(
                "No zone JSON found for video '%s'. Zone-based analysis will be skipped.",
                self.video_path,
            )
            self.zone_data = {}
            self.zone_shapes = []
            self.zone_specs = []
            self.zone_time_dict = {}
            return

        self.zone_file = zone_path
        with open(zone_path, "r") as f:
            self.zone_data = json.load(f)
        logger.info(f"Loading zones from {zone_path}")

        self.zone_specs = load_zone_shapes(self.zone_data)
        self.zone_shapes = [spec.to_shape_dict() for spec in self.zone_specs]
        self.zone_engine = GenericZoneEngine(self.zone_specs, fps=self.fps)
        self.zone_time_dict = {spec.display_label: 0 for spec in self.zone_specs}

    def _resolve_xy_columns(self, dataframe):
        candidates = [
            ("cx_tracked", "cy_tracked"),
            ("cx_tracking", "cy_tracking"),
            ("cx", "cy"),
            ("x", "y"),
        ]
        for x_col, y_col in candidates:
            if x_col in dataframe.columns and y_col in dataframe.columns:
                return x_col, y_col
        raise KeyError(
            "Could not resolve centroid columns. Expected one of "
            "cx_tracked/cy_tracked, cx_tracking/cy_tracking, cx/cy, or x/y."
        )

    def _analysis_dataframe(self):
        if hasattr(self, "merged_df") and self.merged_df is not None:
            return self.merged_df
        if hasattr(self, "tracked_df") and self.tracked_df is not None:
            return self.tracked_df
        if hasattr(self, "tracking_df") and self.tracking_df is not None:
            return self.tracking_df
        self.read_csv_files()
        if hasattr(self, "tracked_df") and self.tracked_df is not None:
            return self.tracked_df
        return self.tracking_df

    def _tracking_observation_dataframe(self) -> pd.DataFrame:
        if hasattr(self, "merged_df") and self.merged_df is not None:
            return self.merged_df
        if hasattr(self, "tracking_df") and self.tracking_df is not None:
            return self.tracking_df
        if hasattr(self, "tracked_df") and self.tracked_df is not None:
            return self.tracked_df
        self.read_csv_files()
        if hasattr(self, "tracking_df") and self.tracking_df is not None:
            return self.tracking_df
        if hasattr(self, "tracked_df") and self.tracked_df is not None:
            return self.tracked_df
        return pd.DataFrame()

    def determine_time_in_zone(self, instance_label):
        """
        Determine the time spent by an instance in each zone.

        Args:
            instance_label (str): The label of the instance.

        Returns:
            dict: A dictionary containing the time spent by the instance in each zone.
        """
        # Filter merged DataFrame for given instance
        dataframe = self._analysis_dataframe()
        if self.zone_engine is None:
            return {}
        profiled_engine = self._engine_for_profile(self.assay_profile)
        return profiled_engine.occupancy_frames_for_instance(dataframe, instance_label)

    def _engine_for_profile(
        self, profile: str | ZoneAssayProfile | None = None
    ) -> GenericZoneEngine:
        resolved = resolve_assay_profile(profile or self.assay_profile)
        filtered_specs = filter_zone_specs_by_assay_profile(self.zone_specs, resolved)
        return GenericZoneEngine(filtered_specs, fps=self.fps)

    def _zone_metrics_dataframe(
        self, profile: str | ZoneAssayProfile | None = None
    ) -> tuple[ZoneAssayProfile, pd.DataFrame]:
        resolved = resolve_assay_profile(profile or self.assay_profile)
        if self.zone_engine is None:
            return resolved, pd.DataFrame()
        analysis_df = self._analysis_dataframe()
        profiled_engine = self._engine_for_profile(resolved)
        summary_df = profiled_engine.summary_dataframe(analysis_df)
        if not summary_df.empty:
            summary_df["assay_profile"] = resolved.name
        return resolved, summary_df

    @staticmethod
    def _computed_metric_names() -> list[str]:
        return [
            "occupancy_frames",
            "occupancy_seconds",
            "dwell_frames",
            "dwell_seconds",
            "entry_counts",
            "transition_counts",
            "barrier_adjacent_frames",
            "barrier_adjacent_seconds",
            "outside_frames",
            "outside_seconds",
            "total_transitions",
        ]

    def _zone_summary_items(
        self, profile: str | ZoneAssayProfile | None = None
    ) -> dict[str, object]:
        resolved = resolve_assay_profile(profile or self.assay_profile)
        included_specs = filter_zone_specs_by_assay_profile(self.zone_specs, resolved)
        included_labels = [spec.display_label for spec in included_specs]
        included_label_set = {label for label in included_labels}
        accessible_specs = [
            spec
            for spec in self.zone_specs
            if str(spec.access_state or "").strip().lower() != "blocked"
        ]
        blocked_specs = [
            spec
            for spec in self.zone_specs
            if str(spec.access_state or "").strip().lower() == "blocked"
        ]
        return {
            "profile": resolved,
            "profile_title": resolved.title,
            "profile_description": resolved.description,
            "total_zones": len(self.zone_specs),
            "profile_included_count": len(included_specs),
            "profile_excluded_count": max(
                0, len(self.zone_specs) - len(included_specs)
            ),
            "accessible_count": len(accessible_specs),
            "blocked_count": len(blocked_specs),
            "included_labels": included_labels,
            "excluded_labels": [
                spec.display_label
                for spec in self.zone_specs
                if spec.display_label not in included_label_set
            ],
            "accessible_labels": [spec.display_label for spec in accessible_specs],
            "blocked_labels": [spec.display_label for spec in blocked_specs],
            "included_specs": included_specs,
            "metric_names": self._computed_metric_names(),
        }

    def _decorate_summary_dataframe(
        self,
        summary_df: pd.DataFrame,
        profile: ZoneAssayProfile,
    ) -> pd.DataFrame:
        if summary_df.empty:
            return summary_df
        zone_stats = self._zone_summary_items(profile)
        summary_df = summary_df.copy()
        summary_df["assay_profile"] = profile.name
        summary_df["profile_title"] = zone_stats["profile_title"]
        summary_df["profile_description"] = zone_stats["profile_description"]
        summary_df["metrics_computed"] = ", ".join(zone_stats["metric_names"])
        summary_df["included_zone_labels"] = ", ".join(zone_stats["included_labels"])
        summary_df["excluded_zone_labels"] = ", ".join(zone_stats["excluded_labels"])
        summary_df["accessible_zone_labels"] = ", ".join(
            zone_stats["accessible_labels"]
        )
        summary_df["blocked_zone_labels"] = ", ".join(zone_stats["blocked_labels"])
        return summary_df

    def _social_zone_specs(
        self, profile: str | ZoneAssayProfile | None = None
    ) -> list[ZoneShapeSpec]:
        resolved = resolve_assay_profile(profile or self.assay_profile)
        profiled_specs = filter_zone_specs_by_assay_profile(self.zone_specs, resolved)
        social_specs: list[ZoneShapeSpec] = []
        for spec in profiled_specs:
            flags = dict(spec.flags or {})
            tags = flags.get("tags")
            tag_text = []
            if isinstance(tags, (list, tuple, set)):
                tag_text = [str(item or "").strip().lower() for item in tags]
            elif isinstance(tags, str):
                tag_text = [part.strip().lower() for part in tags.split(",")]
            label = str(spec.display_label or spec.label or "").strip().lower()
            description = str(spec.description or "").strip().lower()
            if spec.zone_kind in {"doorway", "interaction_zone"}:
                social_specs.append(spec)
            elif {"social_zone", "social", "door_social"} & set(tag_text):
                social_specs.append(spec)
            elif "social" in label or "social" in description:
                social_specs.append(spec)
        return social_specs

    def _load_pose_keypoint_dataframe(self) -> pd.DataFrame:
        try:
            return load_pose_keypoint_observations(self.video_path)
        except Exception:
            logger.debug("Failed to load pose keypoint observations.", exc_info=True)
            return pd.DataFrame()

    def _social_analysis_dataframe(self) -> pd.DataFrame:
        analysis_df = self._tracking_observation_dataframe()
        if analysis_df.empty:
            return analysis_df
        pose_df = self._load_pose_keypoint_dataframe()
        return build_anchor_dataframe(analysis_df, pose_df)

    def _social_summary_items(
        self,
        profile: str | ZoneAssayProfile | None = None,
        *,
        latency_reference_frame: int | None = None,
    ) -> dict[str, object]:
        resolved = resolve_assay_profile(profile or self.assay_profile)
        zone_specs = self._social_zone_specs(resolved)
        return {
            "profile": resolved,
            "profile_title": resolved.title,
            "profile_description": resolved.description,
            "zone_specs": zone_specs,
            "zone_labels": [spec.display_label for spec in zone_specs],
            "latency_reference_frame": latency_reference_frame,
            "latency_reference_text": (
                "first analyzed frame"
                if latency_reference_frame is None
                else f"frame {int(latency_reference_frame)}"
            ),
        }

    def _social_metrics_dataframe(
        self,
        profile: str | ZoneAssayProfile | None = None,
        *,
        latency_reference_frame: int | None = None,
    ) -> tuple[ZoneAssayProfile, pd.DataFrame, pd.DataFrame]:
        resolved = resolve_assay_profile(profile or self.assay_profile)
        social_specs = self._social_zone_specs(resolved)
        if not social_specs:
            return resolved, pd.DataFrame(), pd.DataFrame()

        social_df = self._social_analysis_dataframe()
        if social_df.empty:
            return resolved, pd.DataFrame(), pd.DataFrame()
        if latency_reference_frame is not None:
            try:
                reference_frame = int(latency_reference_frame)
            except Exception:
                reference_frame = 0
            social_df = social_df[social_df["frame_number"] >= reference_frame].copy()
        else:
            reference_frame = 0

        engine = GenericZoneEngine(
            social_specs,
            fps=self.fps,
            point_columns=("anchor_x", "anchor_y"),
        )
        rows: list[dict[str, object]] = []
        for instance_label in social_df["instance_name"].dropna().astype(str).unique():
            result = engine.analyze_instance(social_df, instance_label)
            row = result.to_summary_row(fps=self.fps, include_transition_columns=False)
            instance_df = social_df[
                social_df["instance_name"].astype(str) == instance_label
            ]
            row["anchor_source"] = (
                instance_df["anchor_source"].mode().iat[0]
                if not instance_df.empty
                and "anchor_source" in instance_df.columns
                and not instance_df["anchor_source"].dropna().empty
                else "centroid"
            )
            row["latency_reference_frame"] = reference_frame
            row["latency_reference_mode"] = (
                "frame"
                if latency_reference_frame is not None
                else "first_analyzed_frame"
            )
            for spec in social_specs:
                zone_label = spec.display_label
                row[f"zone_distance_min__{zone_label}"] = None
                row[f"zone_distance_mean__{zone_label}"] = None
                row[f"zone_distance_median__{zone_label}"] = None
                row[f"zone_proximity_mean__{zone_label}"] = None
                row[f"zone_proximity_max__{zone_label}"] = None
            rows.append(row)

        summary_df = pd.DataFrame(rows)
        if summary_df.empty:
            return resolved, pd.DataFrame(), pd.DataFrame()

        # Compute zone distance/proximity metrics using the anchor dataframe.
        for spec in social_specs:
            zone_label = spec.display_label
            distances: list[float] = []
            per_instance_distances: dict[str, list[float]] = {}
            for _, row in social_df.iterrows():
                instance_name = str(row.get("instance_name") or "").strip()
                if not instance_name:
                    continue
                anchor_x = row.get("anchor_x")
                anchor_y = row.get("anchor_y")
                if pd.isna(anchor_x) or pd.isna(anchor_y):
                    continue
                distance = zone_shape_distance_to_point(
                    spec, (float(anchor_x), float(anchor_y))
                )
                if distance is None:
                    continue
                distances.append(distance)
                per_instance_distances.setdefault(instance_name, []).append(distance)

            for instance_name, summary_row in summary_df.set_index(
                "instance_name"
            ).iterrows():
                inst_distances = per_instance_distances.get(instance_name, [])
                if not inst_distances:
                    continue
                mean_distance = float(sum(inst_distances) / len(inst_distances))
                min_distance = float(min(inst_distances))
                median_distance = float(pd.Series(inst_distances).median())
                zone_bounds = zone_shape_bounds(spec)
                if zone_bounds is not None:
                    min_x, min_y, max_x, max_y = zone_bounds
                    zone_scale = max(max_x - min_x, max_y - min_y, 1.0)
                else:
                    zone_scale = 1.0
                proximity_scores = [
                    max(0.0, 1.0 - (distance / zone_scale))
                    for distance in inst_distances
                ]
                summary_df.loc[
                    summary_df["instance_name"] == instance_name,
                    f"zone_distance_min__{zone_label}",
                ] = min_distance
                summary_df.loc[
                    summary_df["instance_name"] == instance_name,
                    f"zone_distance_mean__{zone_label}",
                ] = mean_distance
                summary_df.loc[
                    summary_df["instance_name"] == instance_name,
                    f"zone_distance_median__{zone_label}",
                ] = median_distance
                summary_df.loc[
                    summary_df["instance_name"] == instance_name,
                    f"zone_proximity_mean__{zone_label}",
                ] = float(sum(proximity_scores) / len(proximity_scores))
                summary_df.loc[
                    summary_df["instance_name"] == instance_name,
                    f"zone_proximity_max__{zone_label}",
                ] = float(max(proximity_scores))

                first_entry_column = f"first_entry_frame__{zone_label}"
                first_entry_frame = None
                if first_entry_column in summary_df.columns:
                    first_entry_series = summary_df.loc[
                        summary_df["instance_name"] == instance_name,
                        first_entry_column,
                    ]
                    if not first_entry_series.empty:
                        first_entry_frame = first_entry_series.iloc[0]
                if pd.notna(first_entry_frame):
                    latency_frame = float(first_entry_frame) - float(reference_frame)
                    summary_df.loc[
                        summary_df["instance_name"] == instance_name,
                        f"latency_frame__{zone_label}",
                    ] = latency_frame
                    summary_df.loc[
                        summary_df["instance_name"] == instance_name,
                        f"latency_seconds__{zone_label}",
                    ] = latency_frame / float(self.fps) if self.fps else latency_frame
                else:
                    summary_df.loc[
                        summary_df["instance_name"] == instance_name,
                        f"latency_frame__{zone_label}",
                    ] = None
                    summary_df.loc[
                        summary_df["instance_name"] == instance_name,
                        f"latency_seconds__{zone_label}",
                    ] = None

        summary_df = summary_df.set_index("instance_name")
        pairwise_summary_df, pairwise_detail_df = compute_pairwise_centroid_summary(
            social_df
        )
        if not pairwise_summary_df.empty:
            summary_df = summary_df.join(pairwise_summary_df, how="left")
        summary_df["assay_profile"] = resolved.name
        summary_df["profile_title"] = resolved.title
        summary_df["profile_description"] = resolved.description
        summary_df["latency_reference_text"] = (
            "first analyzed frame"
            if latency_reference_frame is None
            else f"frame {int(latency_reference_frame)}"
        )
        summary_df["metrics_computed"] = ", ".join(
            [
                "occupancy_frames",
                "occupancy_seconds",
                "dwell_frames",
                "dwell_seconds",
                "entry_counts",
                "first_entry_frames",
                "first_entry_seconds",
                "latency_frame",
                "latency_seconds",
                "zone_distance_min",
                "zone_distance_mean",
                "zone_distance_median",
                "zone_proximity_mean",
                "zone_proximity_max",
                "nearest_neighbor_distance",
                "nearest_neighbor_proximity_score",
            ]
        )
        summary_df["social_zone_labels"] = ", ".join(
            f"`{label}`" for label in [spec.display_label for spec in social_specs]
        )
        summary_df["latency_reference_frame"] = reference_frame
        return resolved, summary_df, pairwise_detail_df

    @staticmethod
    def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
        if not rows:
            return ""
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        body_lines = []
        for row in rows:
            body_lines.append("| " + " | ".join(str(item) for item in row) + " |")
        return "\n".join([header_line, separator_line, *body_lines])

    def save_assay_summary_report(
        self,
        output_dir=None,
        assay_profile=None,
    ) -> str | None:
        if not self.zone_specs:
            logger.warning(
                "No zone shapes available for '%s'; skipping assay summary export.",
                self.video_path,
            )
            return None

        resolved_profile, summary_df = self._zone_metrics_dataframe(assay_profile)
        if summary_df.empty:
            logger.warning(
                "No analyzable instance rows found for '%s'; skipping assay summary export.",
                self.video_path,
            )
            return None
        summary_df = self._decorate_summary_dataframe(summary_df, resolved_profile)

        output_root = (
            Path(output_dir) if output_dir is not None else self.video_path.parent
        )
        output_root.mkdir(parents=True, exist_ok=True)
        stem = f"{self.video_path.stem}_{resolved_profile.name}_assay_summary"
        csv_path = output_root / f"{stem}.csv"
        md_path = output_root / f"{stem}.md"

        summary_df.to_csv(csv_path)
        zone_stats = self._zone_summary_items(resolved_profile)
        included_specs = zone_stats["included_specs"]

        with md_path.open("w", encoding="utf-8") as handle:
            handle.write(f"# {resolved_profile.title} Assay Summary\n\n")
            handle.write(f"- **Video:** `{self.video_path}`\n")
            handle.write(f"- **Zone file:** `{self.zone_file or ''}`\n")
            handle.write(f"- **Profile:** `{resolved_profile.name}`\n")
            handle.write(
                f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            handle.write(f"- **CSV output:** `{csv_path.name}`\n\n")
            handle.write(f"{resolved_profile.description}\n\n")

            handle.write("## Phase Rules\n\n")
            phase_rules = [
                ["Profile", resolved_profile.title],
                [
                    "Allowed access states",
                    ", ".join(sorted(resolved_profile.allowed_access_states))
                    if resolved_profile.allowed_access_states
                    else "All non-blocked",
                ],
                [
                    "Included zone kinds",
                    ", ".join(sorted(resolved_profile.included_zone_kinds))
                    if resolved_profile.included_zone_kinds
                    else "All zone kinds",
                ],
                [
                    "Always include zone kinds",
                    ", ".join(sorted(resolved_profile.always_include_zone_kinds)),
                ],
                [
                    "Respect zone phase",
                    "Yes" if resolved_profile.respect_zone_phase else "No",
                ],
            ]
            handle.write(self._markdown_table(["Rule", "Value"], phase_rules))
            handle.write("\n\n")

            handle.write("## Zone Coverage\n\n")
            metrics_rows = [
                ["Total zones", zone_stats["total_zones"]],
                ["Profile-included zones", zone_stats["profile_included_count"]],
                ["Profile-excluded zones", zone_stats["profile_excluded_count"]],
                ["Accessible zones", zone_stats["accessible_count"]],
                ["Blocked zones", zone_stats["blocked_count"]],
            ]
            handle.write(self._markdown_table(["Metric", "Value"], metrics_rows))
            handle.write("\n\n")

            handle.write("## Accessible Zones\n\n")
            accessible_labels = zone_stats["accessible_labels"]
            if accessible_labels:
                handle.write(
                    ", ".join(f"`{label}`" for label in accessible_labels) + "\n\n"
                )
            else:
                handle.write("None\n\n")

            handle.write("## Blocked Zones\n\n")
            blocked_labels = zone_stats["blocked_labels"]
            if blocked_labels:
                handle.write(
                    ", ".join(f"`{label}`" for label in blocked_labels) + "\n\n"
                )
            else:
                handle.write("None\n\n")

            handle.write("## Profile-Included Zones\n\n")
            included_rows = [
                [
                    spec.display_label,
                    spec.zone_kind,
                    spec.access_state,
                    spec.phase,
                ]
                for spec in included_specs
            ]
            handle.write(
                self._markdown_table(
                    ["Label", "Kind", "Access", "Phase"],
                    included_rows,
                )
                or "No zones included by the selected profile.\n"
            )
            handle.write("\n\n")

            handle.write("## Metrics Computed\n\n")
            handle.write(
                ", ".join(f"`{name}`" for name in zone_stats["metric_names"]) + "\n\n"
            )

            handle.write("## Per-Instance Metrics\n\n")
            handle.write(
                "The CSV companion contains one row per instance with the metrics listed above, plus profile metadata and the included/excluded zone lists.\n"
            )

        logger.info(
            "Saving assay summary report to %s and %s using profile %s",
            csv_path,
            md_path,
            resolved_profile.name,
        )
        return str(md_path)

    def plot_time_in_zones(self, instance_label):
        """
        Plot the time spent by an instance in each zone.

        Args:
            instance_label (str): The label of the instance.
        """
        zone_time_dict = self.determine_time_in_zone(instance_label)

        if self.fps:
            plt.bar(
                zone_time_dict.keys(),
                [frames / self.fps for frames in zone_time_dict.values()],
            )
            plt.ylabel("Time (seconds)")
        else:
            plt.bar(zone_time_dict.keys(), zone_time_dict.values())
            plt.ylabel("Time (frames)")
        plt.xlabel("Zone")
        plt.title(f"Time Spent in Each Zone for {instance_label}")
        plt.show()

    def save_all_instances_zone_time_to_csv(self, output_csv=None, assay_profile=None):
        """
        Calculate and save the time spent by each instance in each zone to a CSV file.

        Args:
            output_csv (str): The path to the output CSV file.
        """
        if output_csv is None:
            if self.tracked_csv is None:
                self.tracked_csv = (
                    self.video_path.parent / f"{self.video_path.stem}_tracking.csv"
                )
            output_csv = str(self.tracking_csv).replace(
                "_tracking", "_place_preference"
            )
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile = resolve_assay_profile(assay_profile or self.assay_profile)
        if not self.zone_specs:
            logger.warning(
                "No zone shapes available for '%s'; skipping zone export.",
                self.video_path,
            )
            return output_csv

        analysis_df = self._analysis_dataframe()
        if self.zone_engine is None:
            return output_csv
        profiled_engine = self._engine_for_profile(profile)
        all_instances_zone_time: dict[str, dict[str, float]] = {}
        for instance_label in (
            analysis_df["instance_name"].dropna().astype(str).unique()
        ):
            result = profiled_engine.analyze_instance(analysis_df, instance_label)
            result.assay_profile = profile.name
            all_instances_zone_time[instance_label] = result.occupancy_seconds(self.fps)

        instances_zone_time_df = pd.DataFrame(all_instances_zone_time).T
        if instances_zone_time_df.empty:
            logger.warning(
                "No analyzable instance rows found for '%s'; skipping zone export.",
                self.video_path,
            )
            return output_csv

        instances_zone_time_df.to_csv(output_path)
        logger.info(
            "Saving place preference data to %s using profile %s",
            output_path,
            profile.name,
        )
        return str(output_path)

    def save_zone_metrics_to_csv(self, output_csv=None, assay_profile=None):
        if output_csv is None:
            output_csv = str(self.tracking_csv).replace("_tracking", "_zone_metrics")
        profile, summary_df = self._zone_metrics_dataframe(assay_profile)
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.zone_engine is None:
            logger.warning(
                "No zone engine available for '%s'; skipping zone metrics export.",
                self.video_path,
            )
            return output_csv
        if summary_df.empty:
            logger.warning(
                "No analyzable instance rows found for '%s'; skipping zone metrics export.",
                self.video_path,
            )
            return output_csv
        summary_df = self._decorate_summary_dataframe(summary_df, profile)
        summary_df.to_csv(output_path)
        logger.info(
            "Saving zone metrics data to %s using profile %s",
            output_path,
            profile.name,
        )
        return str(output_path)

    def save_social_metrics_to_csv(
        self,
        output_csv=None,
        assay_profile=None,
        latency_reference_frame: int | None = None,
    ) -> str | None:
        if output_csv is None:
            output_csv = str(self.tracking_csv).replace("_tracking", "_social_metrics")
        profile, summary_df, _ = self._social_metrics_dataframe(
            assay_profile, latency_reference_frame=latency_reference_frame
        )
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if summary_df.empty:
            logger.warning(
                "No analyzable social metrics found for '%s'; skipping export.",
                self.video_path,
            )
            return output_csv
        summary_df.to_csv(output_path)
        logger.info(
            "Saving social metrics data to %s using profile %s",
            output_path,
            profile.name,
        )
        return str(output_path)

    def save_social_summary_report(
        self,
        output_dir=None,
        assay_profile=None,
        latency_reference_frame: int | None = None,
    ) -> str | None:
        if not self.zone_specs:
            logger.warning(
                "No zone shapes available for '%s'; skipping social summary export.",
                self.video_path,
            )
            return None

        profile, summary_df, pairwise_df = self._social_metrics_dataframe(
            assay_profile, latency_reference_frame=latency_reference_frame
        )
        if summary_df.empty:
            logger.warning(
                "No analyzable social metrics found for '%s'; skipping social summary export.",
                self.video_path,
            )
            return None

        output_root = (
            Path(output_dir) if output_dir is not None else self.video_path.parent
        )
        output_root.mkdir(parents=True, exist_ok=True)
        stem = f"{self.video_path.stem}_{profile.name}_social_summary"
        csv_path = output_root / f"{stem}.csv"
        md_path = output_root / f"{stem}.md"
        pairwise_csv_path = output_root / f"{stem}_pairwise.csv"

        summary_df.to_csv(csv_path)
        social_stats = self._social_summary_items(
            profile, latency_reference_frame=latency_reference_frame
        )
        if not pairwise_df.empty:
            pairwise_df.to_csv(pairwise_csv_path, index=False)

        with md_path.open("w", encoding="utf-8") as handle:
            handle.write(f"# {profile.title} Social Summary\n\n")
            handle.write(f"- **Video:** `{self.video_path}`\n")
            handle.write(f"- **Zone file:** `{self.zone_file or ''}`\n")
            handle.write(f"- **Profile:** `{profile.name}`\n")
            handle.write(
                f"- **Latency reference:** `{social_stats['latency_reference_text']}`\n"
            )
            handle.write(
                f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            handle.write(f"- **CSV output:** `{csv_path.name}`\n\n")
            handle.write(
                "Latency is measured from the first analyzed frame unless a reference frame is supplied.\n\n"
            )

            handle.write("## Social Zones\n\n")
            social_rows = [
                [
                    spec.display_label,
                    spec.zone_kind,
                    spec.access_state,
                    spec.phase,
                    spec.description or "",
                ]
                for spec in social_stats["zone_specs"]
            ]
            handle.write(
                self._markdown_table(
                    ["Label", "Kind", "Access", "Phase", "Description"],
                    social_rows,
                )
                or "No social zones were detected for this profile.\n"
            )
            handle.write("\n\n")

            handle.write("## Metrics Computed\n\n")
            handle.write(
                ", ".join(
                    f"`{name}`"
                    for name in [
                        "occupancy_frames",
                        "first_entry_frames",
                        "zone_distance_min",
                        "zone_distance_mean",
                        "zone_distance_median",
                        "zone_proximity_mean",
                        "nearest_neighbor_distance",
                        "nearest_neighbor_proximity_score",
                    ]
                )
                + "\n\n"
            )

            handle.write("## Per-Instance Summary\n\n")
            compact_rows = []
            for _, row in summary_df.reset_index().iterrows():
                compact_rows.append(
                    [
                        row.get("instance_name", ""),
                        row.get("anchor_source", ""),
                        row.get("mean_nearest_neighbor_distance_px", ""),
                        row.get("min_nearest_neighbor_distance_px", ""),
                        row.get("mean_nearest_neighbor_proximity_score", ""),
                        row.get("latency_reference_text", ""),
                    ]
                )
            handle.write(
                self._markdown_table(
                    [
                        "Instance",
                        "Anchor source",
                        "Mean NN distance (px)",
                        "Min NN distance (px)",
                        "Mean NN proximity score",
                        "Latency reference",
                    ],
                    compact_rows,
                )
            )
            handle.write(
                "\n\nThe companion CSV contains the full per-zone latency, dwell, occupancy, and distance columns.\n\n"
            )

            if not pairwise_df.empty:
                handle.write("## Pairwise Centroid Distances\n\n")
                pairwise_rows = []
                for _, row in pairwise_df.head(50).iterrows():
                    pairwise_rows.append(
                        [
                            row.get("frame_number", ""),
                            row.get("instance_name_1", ""),
                            row.get("instance_name_2", ""),
                            row.get("distance_px", ""),
                            row.get("proximity_score", ""),
                        ]
                    )
                handle.write(
                    self._markdown_table(
                        [
                            "Frame",
                            "Instance A",
                            "Instance B",
                            "Distance (px)",
                            "Proximity score",
                        ],
                        pairwise_rows,
                    )
                )
                handle.write(
                    "\n\nThe companion pairwise CSV contains the full table: "
                    f"`{pairwise_csv_path.name}`.\n\n"
                )

        logger.info(
            "Saving social summary report to %s and %s using profile %s",
            csv_path,
            md_path,
            profile.name,
        )
        return str(md_path)


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Track results analyzer")
    parser.add_argument("video_path", type=str, help="Name of the video")
    parser.add_argument(
        "zone_file", type=str, default=None, help="Path to the zone JSON file"
    )
    parser.add_argument("fps", type=float, default=30, help="FPS for the video")
    args = parser.parse_args()

    # Create and run the analyzer
    analyzer = TrackingResultsAnalyzer(
        args.video_path, zone_file=args.zone_file, fps=args.fps
    )
    analyzer.merge_and_calculate_distance()
    time_in_zone_mouse = analyzer.determine_time_in_zone("mouse")
    print("Time in zone for mouse:", time_in_zone_mouse)
    analyzer.plot_time_in_zones("mouse")
    analyzer.save_all_instances_zone_time_to_csv()
