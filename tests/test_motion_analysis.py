import numpy as np
from annolid.postprocessing.motion_analysis import calculate_smoothed_velocity


def test_calculate_smoothed_velocity():
    # Test case 1: Simple linear motion in x direction
    node_locations = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
    expected_velocities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(
        calculate_smoothed_velocity(node_locations, win=5), expected_velocities
    )

    # Test case 2: Simple linear motion in y direction
    node_locations = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]])
    expected_velocities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(
        calculate_smoothed_velocity(node_locations, win=5), expected_velocities
    )

    # Test case 3: Simple diagonal motion
    node_locations = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    expected_velocities = np.sqrt(2) * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(
        calculate_smoothed_velocity(node_locations, win=5), expected_velocities
    )

    # Test case 4: No motion
    node_locations = np.zeros((6, 2))
    expected_velocities = np.zeros(6)
    assert np.allclose(
        calculate_smoothed_velocity(node_locations, win=5), expected_velocities
    )
