import pandas as pd
import json
import itertools
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


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

    def __init__(self, video_name, zone_file):
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

        zone_time_dict = {shape['label']
            : 0 for shape in self.zone_data['shapes']}

        for shape in self.zone_data['shapes']:
            zone_label = shape['label']
            zone_time = 0
            # Check if instance points are within the zone
            for _, row in instance_df.iterrows():
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

        plt.bar(zone_time_dict.keys(), zone_time_dict.values())
        plt.xlabel('Zone')
        plt.ylabel('Time (frames)')
        plt.title(f'Time Spent in Each Zone for {instance_label}')
        plt.show()


if __name__ == '__main__':
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Track results analyzer')
    parser.add_argument('video_name', type=str, help='Name of the video')
    parser.add_argument('zone_file', type=str,
                        help='Path to the zone JSON file')
    args = parser.parse_args()

    # Create and run the analyzer
    analyzer = TrackingResultsAnalyzer(args.video_name, args.zone_file)
    analyzer.merge_and_calculate_distance()
    time_in_zone_mouse = analyzer.determine_time_in_zone("mouse_0")
    print("Time in zone for mouse:", time_in_zone_mouse)
    analyzer.plot_time_in_zones("mouse_0")
