import pandas as pd
import argparse
import os


class FreezingEventAnalyzer:
    """
    A class to analyze freezing events in tracked data from a CSV file.

    Attributes:
        file_path (str): Path to the CSV file containing the tracked data.
        instance_name (str): The name of the instance to analyze.
        threshold (float): The threshold for the motion index to consider as freezing.
        min_duration (int): Minimum duration (in frames) to consider a freezing event.

    Methods:
        load_data(): Loads the CSV data into a DataFrame.
        preprocess_data(): Filters data for the specified instance.
        identify_freezing_events(): Identifies and marks freezing events based on threshold and duration.
        summarize_freezing_events(): Summarizes the freezing events.
        analyze(): Orchestrates the overall analysis process.
    """

    def __init__(self, file_path, instance_name, threshold=0.5, min_duration=10):
        self.file_path = file_path
        self.instance_name = instance_name
        self.threshold = threshold
        self.min_duration = min_duration

    def load_data(self):
        """Loads the CSV data into a DataFrame."""
        self.data = pd.read_csv(self.file_path)
        return self.data

    def preprocess_data(self):
        """Filters and sorts data for the specified instance."""
        filtered = self.data[self.data["instance_name"] == self.instance_name].copy()
        if filtered.empty:
            self.instance_data = filtered
            return
        filtered["frame_number"] = pd.to_numeric(
            filtered["frame_number"], errors="coerce"
        )
        filtered["motion_index"] = pd.to_numeric(
            filtered["motion_index"], errors="coerce"
        )
        filtered = filtered.dropna(subset=["frame_number"])
        if not filtered.empty:
            filtered["frame_number"] = filtered["frame_number"].astype(int)
            filtered = filtered.sort_values("frame_number", kind="mergesort")
            filtered = filtered.drop_duplicates(subset=["frame_number"], keep="last")
        self.instance_data = filtered.reset_index(drop=True)

    def identify_freezing_events(self):
        """Identifies and marks freezing events based on threshold, continuity, and duration."""
        if self.instance_data.empty:
            self.instance_data["is_freezing"] = pd.Series(dtype=bool)
            self.instance_data["freezing_group"] = pd.Series(dtype=int)
            self.instance_data["freezing_event"] = pd.Series(dtype=bool)
            return

        motion = self.instance_data["motion_index"]
        is_freezing = motion.notna() & (motion >= 0.0) & (motion < self.threshold)
        self.instance_data["is_freezing"] = is_freezing

        state_change = is_freezing != is_freezing.shift(
            fill_value=not bool(is_freezing.iloc[0])
        )
        frame_gap = self.instance_data["frame_number"].diff().gt(1).fillna(False)
        self.instance_data["freezing_group"] = (state_change | frame_gap).cumsum()

        freezing_durations = self.instance_data.groupby("freezing_group")[
            "is_freezing"
        ].sum()
        valid_freezing_groups = freezing_durations[
            freezing_durations >= self.min_duration
        ].index
        self.instance_data["freezing_event"] = self.instance_data[
            "freezing_group"
        ].isin(valid_freezing_groups)

    def summarize_freezing_events(self):
        """Summarizes the freezing events."""
        summary_columns = [
            "freezing_group",
            "start_frame",
            "end_frame",
            "start_time",
            "end_time",
        ]
        if (
            self.instance_data.empty
            or "freezing_event" not in self.instance_data.columns
        ):
            return pd.DataFrame(columns=summary_columns)

        active = self.instance_data[self.instance_data["freezing_event"]]
        if active.empty:
            return pd.DataFrame(columns=summary_columns)

        time_col = None
        for candidate in ("timestamps", "timestamp", "time"):
            if candidate in active.columns:
                time_col = candidate
                break

        if time_col is not None:
            freezing_events_summary = active.groupby(
                "freezing_group", as_index=False
            ).agg(
                start_frame=("frame_number", "min"),
                end_frame=("frame_number", "max"),
                start_time=(time_col, "min"),
                end_time=(time_col, "max"),
            )
        else:
            freezing_events_summary = active.groupby(
                "freezing_group", as_index=False
            ).agg(
                start_frame=("frame_number", "min"),
                end_frame=("frame_number", "max"),
            )
            freezing_events_summary["start_time"] = None
            freezing_events_summary["end_time"] = None

        return freezing_events_summary[summary_columns]

    def analyze(self):
        """Orchestrates the overall analysis process."""
        self.load_data()
        self.preprocess_data()
        self.identify_freezing_events()
        summary = self.summarize_freezing_events()
        return summary


def main(file_path, instance_name, output_path=None, threshold=0.5, min_duration=10):
    """
    Main function to run the analysis from the command line.

    Args:
        file_path (str): Path to the CSV file containing the tracked data.
        instance_name (str): The name of the instance to analyze.
        output_path (str): Path to save the CSV file with the summarized results (default is None).
        threshold (float): The threshold for the motion index to consider as freezing (default is 0.5).
        min_duration (int): Minimum duration (in frames) to consider a freezing event (default is 10).
    """
    if output_path is None:
        base_name = os.path.splitext(file_path)[0]
        output_path = f"{base_name}_freezing_summary.csv"

    analyzer = FreezingEventAnalyzer(file_path, instance_name, threshold, min_duration)
    summary = analyzer.analyze()
    summary.to_csv(output_path, index=False)
    print(f"Freezing events summary saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze freezing events in tracked data from a CSV file."
    )
    parser.add_argument(
        "file_path", type=str, help="Path to the CSV file containing the tracked data."
    )
    parser.add_argument(
        "instance_name", type=str, help="The name of the instance to analyze."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the CSV file with the summarized results (default is None).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="The threshold for the motion index to consider as freezing (default is 0.5).",
    )
    parser.add_argument(
        "--min_duration",
        type=int,
        default=10,
        help="Minimum duration (in frames) to consider a freezing event (default is 10).",
    )

    args = parser.parse_args()
    main(
        args.file_path,
        args.instance_name,
        args.output_path,
        args.threshold,
        args.min_duration,
    )
