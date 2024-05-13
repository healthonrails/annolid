import os
import pandas as pd
import json
import itertools
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from annolid.utils.files import find_manual_labeled_json_files


class TrackingResultsAnalyzer:
    """
    A class to analyze tracking results
      and visualize time spent in zones for instances.

    Attributes:
        video_name (str): The name of the video.
        zone_file (str): The path to the JSON file containing zone information.
        tracking_csv (str): The path to the tracking CSV file.
        tracked_csv (str): The path to the tracked CSV file.
        tracking_df (DataFrame): DataFrame containing tracking data.
        tracked_df (DataFrame): DataFrame containing tracked data.
        merged_df (DataFrame): DataFrame containing merged tracking and tracked data.
        distances_df (DataFrame): DataFrame containing distances between instances.
        zone_data (dict): Dictionary containing zone information loaded from the zone JSON file.
    """

    def __init__(self, video_name, zone_file=None, fps=None):
        """
        Initialize the TrackingResultsAnalyzer.

        Args:
            video_name (str): The name of the video.
            zone_file (str): The path to the JSON file containing zone information.
        """
        self.video_name = video_name
        self.tracking_csv = f"{video_name}_tracking.csv"
        self.tracked_csv = f"{video_name}_tracked.csv"
        self.zone_file = zone_file
        self.fps = fps

    def read_csv_files(self):
        """Read tracking and tracked CSV files into DataFrames."""
        self.tracking_df = pd.read_csv(self.tracking_csv)
        self.tracked_df = pd.read_csv(self.tracked_csv)

    def merge_and_calculate_distance(self):
        """Merge tracking and tracked dataframes based on
          frame number and instance name, and calculate distances."""
        self.read_csv_files()

        # Merge DataFrames based on frame number and instance name
        self.merged_df = pd.merge(self.tracking_df, self.tracked_df,
                                  on=['frame_number', 'instance_name'],
                                  suffixes=('_tracking', '_tracked'))

        # Calculate distance between different instances in the same frame
        distances = []
        for frame_number, frame_group in self.merged_df.groupby('frame_number'):
            instances_in_frame = frame_group['instance_name'].unique()
            instance_combinations = itertools.combinations(
                instances_in_frame, 2)
            for instance_combination in instance_combinations:
                instance1 = instance_combination[0]
                instance2 = instance_combination[1]
                instance1_data = frame_group[frame_group['instance_name'] == instance1]
                instance2_data = frame_group[frame_group['instance_name'] == instance2]
                for _, row1 in instance1_data.iterrows():
                    for _, row2 in instance2_data.iterrows():
                        distance = self.calculate_distance(row1['cx_tracking'],
                                                           row1['cy_tracking'],
                                                           row2['cx_tracked'],
                                                           row2['cy_tracked'])
                        distances.append({
                            'frame_number': frame_number,
                            'instance_name_1': instance1,
                            'instance_name_2': instance2,
                            'distance': distance
                        })

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
        if not os.path.exists(self.zone_file):
            json_files = sorted(find_manual_labeled_json_files(
                self.tracked_csv.replace('_tracking.csv', '')))
            # assume the first file has the Zone or place info
            self.zone_file = json_files[0]
        with open(self.zone_file, 'r') as f:
            self.zone_data = json.load(f)

    def determine_time_in_zone(self, instance_label):
        """
        Determine the time spent by an instance in each zone.

        Args:
            instance_label (str): The label of the instance.

        Returns:
            dict: A dictionary containing the time spent by the instance in each zone.
        """
        self.load_zone_json()

        # Filter merged DataFrame for given instance
        instance_df = self.merged_df[self.merged_df['instance_name']
                                     == instance_label]

        zone_shapes = [zone_shape for zone_shape in self.zone_data['shapes']
                       if 'description' in zone_shape and 'zone' in zone_shape['description'].lower()
                       or 'zone' in zone_shape['label'].lower()]
        zone_time_dict = {shape['label']: 0 for shape in zone_shapes}

        for shape in zone_shapes:
            zone_label = shape['label']
            zone_time = 0
            # Check if instance points are within the zone
            for _, row in instance_df.iterrows():
                if len(shape['points']) > 3:
                    if self.is_point_in_polygon([row['cx_tracked'],
                                                row['cy_tracked']], shape['points']):
                        zone_time += 1

            zone_time_dict[zone_label] = zone_time

        return zone_time_dict

    def is_point_in_polygon(self, point, polygon_points):
        """
        Check if a point is inside a polygon.

        Args:
            point (tuple): The coordinates of the point (x, y).
            polygon_points (list): List of tuples representing the polygon vertices.

        Returns:
            bool: True if the point is inside the polygon, False otherwise.
        """
        # Create a Shapely Point object
        point = Point(point[0], point[1])

        # Create a Shapely Polygon object
        polygon = Polygon(polygon_points)

        # Check if the point is within the polygon
        return polygon.contains(point)

    def plot_time_in_zones(self, instance_label):
        """
        Plot the time spent by an instance in each zone.

        Args:
            instance_label (str): The label of the instance.
        """
        zone_time_dict = self.determine_time_in_zone(instance_label)

        if self.fps:
            plt.bar(zone_time_dict.keys(), [
                    frames/self.fps for frames in zone_time_dict.values()])
            plt.ylabel('Time (seconds)')
        else:
            plt.bar(zone_time_dict.keys(), zone_time_dict.values())
            plt.ylabel('Time (frames)')
        plt.xlabel('Zone')
        plt.title(f'Time Spent in Each Zone for {instance_label}')
        plt.show()

    def save_all_instances_zone_time_to_csv(self, output_csv=None):
        """
        Calculate and save the time spent by each instance in each zone to a CSV file.

        Args:
            output_csv (str): The path to the output CSV file.
        """
        if output_csv is None:
            output_csv = self.tracking_csv.replace(
                '_tracking', '_place_preference')

        # Initialize dictionary to store zone time results for all instances
        all_instances_zone_time = {}

        # Iterate over all instances in the tracking dataframe
        for instance_label in self.tracking_df['instance_name'].unique():
            zone_time_dict = self.determine_time_in_zone(instance_label)
            # Convert time from frames to seconds
            if self.fps:
                zone_time_dict = {
                    zone_label: frames/self.fps for zone_label, frames in zone_time_dict.items()}
            all_instances_zone_time[instance_label] = zone_time_dict

        # Convert the dictionary to a DataFrame
        instances_zone_time_df = pd.DataFrame(all_instances_zone_time).T

        # Save the DataFrame to a CSV file
        instances_zone_time_df.to_csv(output_csv)


if __name__ == '__main__':
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Track results analyzer')
    parser.add_argument('video_name', type=str, help='Name of the video')
    parser.add_argument('zone_file', type=str, default=None,
                        help='Path to the zone JSON file')
    parser.add_argument('fps', type=float, default=30,
                        help='FPS for the video')
    args = parser.parse_args()

    # Create and run the analyzer
    analyzer = TrackingResultsAnalyzer(
        args.video_name, zone_file=args.zone_file, fps=args.fps)
    analyzer.merge_and_calculate_distance()
    time_in_zone_mouse = analyzer.determine_time_in_zone("mouse")
    print("Time in zone for mouse:", time_in_zone_mouse)
    analyzer.plot_time_in_zones("mouse")
    analyzer.save_all_instances_zone_time_to_csv()
