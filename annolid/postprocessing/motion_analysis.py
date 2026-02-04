import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def df_to_center_loc(tracking_csv):
    df = pd.read_csv(tracking_csv)
    # pivot the DataFrame to have a hierarchical column index with instance_name on the top level
    df_pivot = pd.pivot_table(
        df, values=["cx", "cy"], index=["frame_number"], columns=["instance_name"]
    )

    # convert the pivoted DataFrame to a numpy array
    center_loc = np.array(df_pivot)

    # reshape the array to the desired shape
    center_loc = center_loc.reshape((center_loc.shape[0], -1, 2))

    # return the reshaped array
    return center_loc


def plot_center_locations(center_loc, labels=None):
    import seaborn as sns
    import matplotlib as mpl

    """
    Plots the center locations of multiple objects over time, and the tracks of the objects.

    Args:
    - center_loc (numpy.ndarray): An array of shape (num_frames, 2, num_objects) containing the x and y
                                  coordinates of center position data for multiple objects.
    - labels (list of str, optional): A list of strings containing labels for each object.

    Returns:
    - figs (list of matplotlib.figure.Figure): The matplotlib figure objects containing the plots.
    """

    num_objects = center_loc.shape[2]

    if labels is None:
        labels = [f"object-{i}" for i in range(num_objects)]

    sns.set("notebook", "ticks", font_scale=1.2)
    mpl.rcParams["figure.figsize"] = [15, 6]

    fig1, ax1 = plt.subplots()
    for i in range(num_objects):
        ax1.plot(center_loc[:, 0, i], color=f"C{i}", label=labels[i])
        ax1.plot(-1 * center_loc[:, 1, i], color=f"C{i}")
    ax1.legend(loc="center right")
    ax1.set_title("Center locations")

    fig2, ax2 = plt.subplots(figsize=(7, 7))
    for i in range(num_objects):
        ax2.plot(
            center_loc[:, 0, i], center_loc[:, 1, i], color=f"C{i}", label=labels[i]
        )
    ax2.legend()
    ax2.set_xlim(0, 1024)
    ax2.set_xticks([])
    ax2.set_ylim(0, 1024)
    ax2.set_yticks([])
    ax2.set_title("Center tracks")

    figs = [fig1, fig2]

    return figs


def calculate_correlation(data_x, data_y, window_size):
    """
    Calculates the rolling correlation between two time series data_x
    and data_y with a window size of window_size.

    Parameters:
    data_x (numpy array): First time series
    data_y (numpy array): Second time series
    window_size (int): Size of the rolling window used for correlation calculation

    Returns:
    numpy array: Array of rolling correlation values
    """

    series_x = pd.Series(data_x)
    series_y = pd.Series(data_y)

    return np.array(series_y.rolling(window_size).corr(series_x))


def plot_center_tracks_and_velocities(center_loc, center_vel):
    """
    Plots the tracks of instances and colors them by the magnitude of their velocity.

    Args:
    - center_loc: numpy array of shape (num_frames, num_instances,
      2) representing the center locations of instances
    - center_vel: numpy array of shape (num_frames, num_instances,
    2) representing the velocity of instances

    Returns:
    - None
    """

    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(121)
    ax1.plot(center_loc[:, 0], center_loc[:, 1])
    ax1.set_xlim(0, 1024)
    ax1.set_xticks([])
    ax1.set_ylim(0, 1024)
    ax1.set_yticks([])
    ax1.set_title("Center tracks")

    kp = center_vel
    vmin = 0
    vmax = 10

    ax2 = fig.add_subplot(122)
    ax2.scatter(center_loc[:, 0], center_loc[:, 1], c=kp, s=4, vmin=vmin, vmax=vmax)
    ax2.set_xlim(0, 1024)
    ax2.set_xticks([])
    ax2.set_ylim(0, 1024)
    ax2.set_yticks([])
    ax2.set_title("Center tracks colored by magnitude of velocity")
    ax2.legend()

    plt.show()


def plot_covariance(center_vel1, center_vel2):
    """
    Plots the covariance of two velocity vectors

    Args:
        center_vel1 (ndarray): velocity vector for the first instance with shape (n_frames,)
        center_vel2 (ndarray): velocity vector for the second instance with shape (n_frames,)

    Returns:
        None
    """

    cov_vel = calculate_correlation(center_vel1, center_vel2, window_size=50)

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 6))
    ax[0].plot(center_vel1, "y", label="Instance 1")
    ax[0].plot(center_vel2, "g", label="Instance 2")
    ax[0].legend()
    ax[0].set_title("Forward Velocity")

    ax[1].plot(cov_vel, "c", markersize=1)
    ax[1].set_ylim(-1.2, 1.2)
    ax[1].set_title("Covariance")

    fig.tight_layout()
    plt.show()


def plot_center_velocities(center_loc, center_vel):
    fig = plt.figure(figsize=(15, 7))

    # plot the x and y positions of the center locations for the first instance
    ax1 = fig.add_subplot(211)
    ax1.plot(center_loc[:, 0, 0], "k", label="x")
    ax1.plot(-1 * center_loc[:, 1, 0], "k", label="y")
    ax1.legend()
    ax1.set_xticks([])
    ax1.set_title("Center Positions")

    # plot the velocity of the first instance
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.imshow(center_vel[:, np.newaxis].T, aspect="auto", vmin=0, vmax=10)
    ax2.set_yticks([])
    ax2.set_title("Velocity")

    plt.show()


def calculate_smoothed_velocity(
    node_locations: np.ndarray, win: int = 25, poly: int = 3
) -> np.ndarray:
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
        node_locations, window_length=win, polyorder=poly, deriv=1, axis=0
    )

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
    instances = df.groupby("instance_name")

    objects = []

    # Loop over instances
    for instance_name, instance in instances:
        # Sort by frame number
        instance = instance.sort_values(by="frame_number")

        # Loop over frames
        for i in range(1, len(instance)):
            # Get instance data for current and previous frames
            curr_row = instance.iloc[i]
            prev_row = instance.iloc[i - 1]

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
                objects.append(
                    {
                        "instance_name": instance_name,
                        "frame_start": prev_row.frame_number,
                        "frame_end": curr_row.frame_number,
                        "box_start": prev_box,
                        "box_end": curr_box,
                        "speed": speed,
                        "direction": direction,
                        "distance": distance,
                    }
                )

        if save_path:
            df = pd.DataFrame(objects)
            df.to_csv(save_path, index=False)

    return objects


if __name__ == "__main__":
    tracking_csv = "mask_rcnn_tracking_results_with_segmenation.csv"
    fps = 30
    motions = calculate_object_motion(
        tracking_csv, fps, tracking_csv.replace(".csv", "_speed.csv")
    )
