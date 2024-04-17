import os
import h5py
from annolid.annotation.keypoints import save_labels
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from annolid.gui.shape import Shape
from annolid.data.videos import CV2Video


def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first.
    Reference: https://sleap.ai/notebooks/Analysis_examples.html
    """

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(
            mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y


def plot_keypoint_locations(filename, keypoint_name):
    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]

    # Create KEYPOINT_MAP from node_names
    KEYPOINT_MAP = {name.lower(): i for i, name in enumerate(node_names)}

    assert keypoint_name.lower() in KEYPOINT_MAP, "Invalid keypoint name. Supported keypoints: " + \
        ", ".join(KEYPOINT_MAP.keys())
    keypoint_index = KEYPOINT_MAP[keypoint_name.lower()]

    print("===filename===")
    print(filename)
    print()

    print("===HDF5 datasets===")
    print(dset_names)
    print()

    print("===locations data shape===")
    print(locations.shape)
    print()

    print("===nodes===")
    for i, name in enumerate(node_names):
        print(f"{i}: {name}")
    print()

    frame_count, node_count, _, instance_count = locations.shape

    print("frame count:", frame_count)
    print("node count:", node_count)
    print("instance count:", instance_count)

    locations = fill_missing(locations)

    keypoint_loc = locations[:, keypoint_index, :, :]

    plt.figure()
    plt.plot(keypoint_loc[:, 0, 0], 'y', label=f"{keypoint_name}-0")
    plt.plot(-1 * keypoint_loc[:, 1, 0], 'y')
    plt.legend(loc="center right")
    plt.title(f"{keypoint_name.capitalize()} locations")

    plt.figure(figsize=(7, 7))
    plt.plot(keypoint_loc[:, 0, 0], keypoint_loc[:, 1, 0],
             'y', label=f"{keypoint_name}-0")
    plt.legend()
    plt.xlim(0, 1024)
    plt.xticks([])
    plt.ylim(0, 1024)
    plt.yticks([])
    plt.title(f"{keypoint_name.capitalize()} tracks")

    plt.show()


def plot_all_keypoints(filename):
    """
    Plot the locations and tracks of all keypoints in the provided HDF5 file.

    Parameters:
        filename (str): Path to the HDF5 file containing the keypoints data.

    Returns:
        None
    """
    with h5py.File(filename, "r") as f:
        # Extract datasets
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]

    # Create KEYPOINT_MAP from node_names
    KEYPOINT_MAP = {name.lower(): i for i, name in enumerate(node_names)}

    print("===filename===")
    print(filename)
    print()

    print("===HDF5 datasets===")
    print(dset_names)
    print()

    print("===nodes===")
    for i, name in enumerate(node_names):
        print(f"{i}: {name}")
    print()

    frame_count, node_count, _, instance_count = locations.shape

    print("===locations data shape===")
    print(locations.shape)
    print()

    print("frame count:", frame_count)
    print("node count:", node_count)
    print("instance count:", instance_count)

    locations = fill_missing(locations)

    # Plot each keypoint
    plt.figure(figsize=(15, 10))
    for keypoint_name, keypoint_index in KEYPOINT_MAP.items():
        keypoint_loc = locations[:, keypoint_index, :, :]
        plt.plot(keypoint_loc[:, 0, 0], keypoint_loc[:, 1, 0],
                 label=f"{keypoint_name.capitalize()}-0")

    plt.legend()
    plt.xlim(0, 1024)
    plt.ylim(0, 1024)
    plt.title('All Keypoints Tracks')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()


def get_frame_info(video_file=None):
    """
    Get frame information (height, width, and total number of frames) from a video file.

    Parameters:
        video_file (str, optional): Path to the video file. Defaults to None.

    Returns:
        tuple: A tuple containing height, width, and number of frames.
    """
    if video_file is not None:
        video_loader = CV2Video(video_file)
        first_frame = video_loader.get_first_frame()
        height = video_loader.get_height()
        width = video_loader.get_width()
        num_frames = video_loader.total_frames()
    else:
        height, width, num_frames = 600, 800, 89761
    return height, width, num_frames


def convert_sleap_h5_to_labelme(h5_file_path):
    """
    Convert a SLEAP HDF5 file to Labelme JSON files.

    Parameters:
        h5_file_path (str): Path to the SLEAP HDF5 file.

    Returns:
        None
    """
    # Output folder name without extension
    output_folder = os.path.splitext(h5_file_path)[0]
    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)
    video_file = h5_file_path.replace('.h5', 'mp4')
    if os.path.exists(video_file):
        height, width, _ = get_frame_info(video_file)
    else:
        height, width, _ = 600, 800, 89761

    # Determine image information
    video_name = os.path.splitext(os.path.basename(h5_file_path))[0]

    with h5py.File(h5_file_path, 'r') as f:
        # Extract relevant datasets
        locations = f["tracks"][:].T
        locations = fill_missing(locations)
        node_names = [n.decode() for n in f["node_names"][:]]

        # Get the dimensions from the data
        frame_count, node_count, _, instance_count = locations.shape

        # Iterate through frames
        for frame_idx in range(frame_count):
            shape_list = []
            for instance_idx in range(instance_count):
                # Iterate through nodes in the instance

                for node_idx, node_name in enumerate(node_names):
                    # Extract coordinates
                    x, y = locations[frame_idx, node_idx, :, instance_idx]

                    # Create Labelme shape
                    shape = Shape(label=node_name,
                                  shape_type='point',
                                  group_id=None,
                                  flags={},
                                  visible=True
                                  )
                    shape.points = [[x, y]]
                    # Add shape to annotation
                    shape_list.append(shape)

            json_file = os.path.join(
                output_folder, f"{video_name}_{frame_idx:0>{9}}.json")
            img_path = json_file.replace('.json', '.png')
            save_labels(json_file, img_path, shape_list, height, width)
            if frame_idx % 100 == 0:
                print(f"Saving file {json_file}")


if __name__ == '__main__':
    # Example usage
    # plot_keypoint_locations("/Downloads/R2311_P4S1_reencoded.h5",
    #                        keypoint_name='head')
    # plot_all_keypoints("/Downloads/R2311_P4S1_reencoded.h5")
    # plt.show()
    convert_sleap_h5_to_labelme(
        "/Downloads/R2311_P4S1_reencoded.h5")
