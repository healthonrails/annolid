import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


"""
The specific features that research papers report for motion analysis depend on 
the particular application and research question. 
However, some common features that are often reported in motion analysis include:

Speed: The speed of the object's motion is one of the most basic features and 
is often reported in terms of distance traveled per unit time 
(e.g., meters per second or kilometers per hour).

Direction: The direction of the object's motion is often reported as an angle 
relative to a reference axis (e.g., degrees or radians).

Acceleration: The acceleration of an object is the rate at which its 
velocity changes over time. This is often reported in terms of the
 change in speed or direction over time.

Trajectory: The trajectory of an object is the path that it follows 
as it moves through space. This can be visualized using plots 
or maps to show the object's position over time.

Frequency: The frequency of an object's motion is the number of cycles
 or repetitions of a motion pattern that occur in a given period of time.

Amplitude: The amplitude of an object's motion is the extent or magnitude 
of its displacement from a reference point or position.

Periodicity: Periodicity refers to the regularity or predictability of 
an object's motion over time. This can be measured using statistical
 analysis techniques such as Fourier analysis.

Spatial patterns: The spatial patterns of an object's motion refer 
to the way in which it moves through space. This can include patterns
 such as oscillations, spirals, or random walks.

Interaction with other objects: The way in which an object interacts
 with other objects in its environment can also be an important feature 
 to report in motion analysis. This can include features such as collisions,
   avoidance behaviors, or interactions with obstacles.

These are just some of the common features that are often reported in motion
 analysis research papers. The specific features that are relevant will 
 depend on the particular application and research question.
"""


def calculate_smoothed_velocity(node_locations: np.ndarray,
                                win: int = 25,
                                poly: int = 3) -> np.ndarray:
    """
    Calculate the velocity of a time series data for each coordinate of a node location array,
    after smoothing it with the Savitzky-Golay filter.

    Parameters
    ----------
    node_locations : np.ndarray
        A 2D array with shape (frames, 2) representing the node locations over time.
    win : int, optional
        The length of the window to use for smoothing. Default is 25.
    poly : int, optional
        The order of the polynomial to fit with. Default is 3.

    Returns
    -------
    np.ndarray
        A 1D array representing the magnitude of the first derivative of the smoothed data.
    """
    # Apply Savitzky-Golay filter to the input data to smooth it
    node_locations_smoothed = savgol_filter(
        node_locations, window_length=win, polyorder=poly, deriv=1, axis=0)

    # Calculate the velocity by taking the norm of the first derivative of the smoothed data in each dimension
    node_velocities = np.linalg.norm(node_locations_smoothed, axis=1)

    return node_velocities


def calculate_object_motion(csv_path, fps, save_path):
    """
    Calculates the motion parameters (speed, direction, 
    traveling distance) for each tracked object in a video.

    Arguments:
    csv_path -- the path to a CSV file containing object
      detections and segmentation masks for each frame
    fps -- the frame rate of the video

    Returns:
    objects -- a list of dictionaries containing
      the motion parameters for each tracked object
    """
    # Load CSV file
    df = pd.read_csv(csv_path)

    # Group by instance name
    instances = df.groupby('instance_name')

    objects = []

    # Loop over instances
    for instance_name, instance in instances:
        # Sort by frame number
        instance = instance.sort_values(by='frame_number')

        # Loop over frames
        for i in range(1, len(instance)):
            # Get instance data for current and previous frames
            curr_row = instance.iloc[i]
            prev_row = instance.iloc[i-1]

            # Check if instance exists in both frames
            if curr_row.instance_name == prev_row.instance_name:
                # Calculate motion parameters
                curr_box = [curr_row.x1, curr_row.y1, curr_row.x2, curr_row.y2]
                prev_box = [prev_row.x1, prev_row.y1, prev_row.x2, prev_row.y2]
                displacement = np.array(curr_box[:2]) - np.array(prev_box[:2])
                distance = np.linalg.norm(displacement)
                time_diff = 1 / fps
                speed = distance / time_diff
                direction = np.arctan2(displacement[1], displacement[0])

                # Add object to list
                objects.append({'instance_name': instance_name,
                                'frame_start': prev_row.frame_number,
                                'frame_end': curr_row.frame_number,
                               'box_start': prev_box, 'box_end': curr_box,
                                'speed': speed, 'direction': direction,
                                'distance': distance})

        if save_path:
            df = pd.DataFrame(objects)
            df.to_csv(save_path, index=False)

    return objects


if __name__ == '__main__':
    tracking_csv = 'mask_rcnn_tracking_results_with_segmenation.csv'
    fps = 30
    motions = calculate_object_motion(
        tracking_csv, fps, tracking_csv.replace('.csv', '_speed.csv'))
