"""
Modified from here: 
https://github.com/nwojke/deep_sort/blob/master/deep_sort/kalman_filter.py
"""

import numpy as np
import scipy.linalg


class KalmanFilter(object):
    """
    A Kalman filter for tracking bbox. 
    Assumption: constant velocity model
    x, y, a, h, vx, vy, va, vh
    """

    def __init__(self):
        n_dim = 4
        dt = 1.

        # Create Kalman filter model matrices.
        # Constanct Velocity Model
        self.motion_matrix = np.eye(2 * n_dim, 2 * n_dim)
        for i in range(n_dim):
            self.motion_matrix[i, n_dim + i] = dt
        self.update_matrix = np.eye(n_dim, 2 * n_dim)

        self.std_weight_position = 1. / 10
        self.std_weight_velocity = 1. / 80

    def initiate(self, measurement):
        """
        measurement is a bbox
        """
        mean_position = measurement
        mean_velocity = np.zeros_like(mean_position)
        mean = np.r_[mean_position, mean_velocity]
        std = [
            2 * self.std_weight_position * measurement[3],
            2 * self.std_weight_position * measurement[3],
            1e-2,
            2 * self.std_weight_position * measurement[3],
            10 * self.std_weight_velocity * measurement[3],
            10 * self.std_weight_velocity * measurement[3],
            1e-5,
            10 * self.std_weight_velocity * measurement[3]
        ]

        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_position = [
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-2,
            self.std_weight_position * mean[3]
        ]
        std_velocity = [
            self.std_weight_velocity * mean[3],
            self.std_weight_velocity * mean[3],
            1e-5,
            self.std_weight_velocity * mean[3]
        ]

        # init Q
        motion_covariance = np.diag(np.square(
            np.r_[std_position, std_velocity]
        ))
        # X' = Fx
        mean = np.dot(self.motion_matrix, mean)

        # P' = FPF.T + Q
        covariance = np.linalg.multi_dot((
            self.motion_matrix,
            covariance,
            self.motion_matrix.T
        )) + motion_covariance

        return mean, covariance

    def update(self,
               mean,
               covariance,
               measurement
               ):
        prejected_mean, projected_covariance = self.predict(mean,
                                                            covariance)
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_covariance,
            lower=True,
            check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self.update_matrix.T).T,
            check_finite=False
        ).T
        innovation = measurement - prejected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (
                kalman_gain,
                projected_covariance,
                kalman_gain.T
            ))
        return new_mean, new_covariance

    def project(self, mean, covariance):
        """
        project state distribution to measurement space.
        """
        std = [
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-1,
            self.std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self.update_matrix, mean)
        covariance = np.linalg.multi_dot((
            self.update_matrix,
            covariance,
            self.update_matrix.T
        ))

        return mean, covariance + innovation_cov

    def gating_distance(self,
                        mean,
                        covariance,
                        measurements,
                        only_position=False
                        ):
        """
        Gating distance between state distribution 
        and mesurements. 
    
        """

        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor,
            d.T,
            lower=True,
            check_finite=False,
            overwrite_b=True
        )
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
