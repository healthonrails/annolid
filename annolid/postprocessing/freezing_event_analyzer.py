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
        """Filters data for the specified instance."""
        self.instance_data = self.data[self.data['instance_name']
                                       == self.instance_name].copy()

    def identify_freezing_events(self):
        """Identifies and marks freezing events based on threshold and duration."""
        self.instance_data['is_freezing'] = self.instance_data['motion_index'] < self.threshold
        self.instance_data['freezing_group'] = (
            self.instance_data['is_freezing'] != self.instance_data['is_freezing'].shift()).cumsum()
        freezing_durations = self.instance_data.groupby(
            'freezing_group').agg({'is_freezing': 'sum'})
        valid_freezing_groups = freezing_durations[freezing_durations['is_freezing']
                                                   >= self.min_duration].index
        self.instance_data['freezing_event'] = self.instance_data['freezing_group'].apply(
            lambda x: x in valid_freezing_groups)

    def summarize_freezing_events(self):
        """Summarizes the freezing events."""
        freezing_events_summary = self.instance_data[self.instance_data['freezing_event']].groupby('freezing_group').agg({
            'frame_number': ['min', 'max'],
            'timestamps': ['min', 'max']
        }).reset_index()
        freezing_events_summary.columns = [
            'freezing_group', 'start_frame', 'end_frame', 'start_time', 'end_time']
        return freezing_events_summary

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

    analyzer = FreezingEventAnalyzer(
        file_path, instance_name, threshold, min_duration)
    summary = analyzer.analyze()
    summary.to_csv(output_path, index=False)
    print(f"Freezing events summary saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze freezing events in tracked data from a CSV file.")
    parser.add_argument('file_path', type=str,
                        help='Path to the CSV file containing the tracked data.')
    parser.add_argument('instance_name', type=str,
                        help='The name of the instance to analyze.')
    parser.add_argument('--output_path', type=str,
                        help='Path to save the CSV file with the summarized results (default is None).')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='The threshold for the motion index to consider as freezing (default is 0.5).')
    parser.add_argument('--min_duration', type=int, default=10,
                        help='Minimum duration (in frames) to consider a freezing event (default is 10).')

    args = parser.parse_args()
    main(args.file_path, args.instance_name,
         args.output_path, args.threshold, args.min_duration)
