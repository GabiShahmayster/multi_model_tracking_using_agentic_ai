"""
Kalman Filter Trackers for 2D Motion Tracking

This module implements three different linear Kalman filter trackers for tracking
objects in 2D space using only position measurements:

1. StaticKalmanFilter: For stationary targets (position only)
2. ConstantVelocityKalmanFilter: For targets moving with constant velocity
3. ConstantAccelerationKalmanFilter: For targets with constant acceleration

All trackers use position-only measurements and estimate the full state
(position, velocity, acceleration as applicable).
"""

import numpy as np
import unittest
from typing import Tuple, Optional


class StaticKalmanFilter:
    """
    Kalman filter for tracking stationary targets.
    
    State vector: [x, y] - position only
    Measurement: [x, y] - position measurements
    
    Assumes the target remains stationary with some process noise.
    """
    
    def __init__(self, initial_position: Tuple[float, float], 
                 position_uncertainty: float = 1.0,
                 process_noise: float = 0.1,
                 measurement_noise: float = 1.0):
        """
        Initialize the static Kalman filter.
        
        Args:
            initial_position: Initial (x, y) position estimate
            position_uncertainty: Initial position uncertainty (standard deviation)
            process_noise: Process noise (how much the position can drift)
            measurement_noise: Measurement noise (measurement uncertainty)
        """
        # State vector: [x, y]
        self.x = np.array([[initial_position[0]], [initial_position[1]]], dtype=float)
        
        # State covariance matrix
        self.P = np.eye(2) * (position_uncertainty ** 2)
        
        # State transition matrix (identity - no motion)
        self.F = np.eye(2)
        
        # Process noise covariance matrix
        self.Q = np.eye(2) * (process_noise ** 2)
        
        # Measurement matrix (observe position directly)
        self.H = np.eye(2)
        
        # Measurement noise covariance matrix
        self.R = np.eye(2) * (measurement_noise ** 2)
    
    def predict(self, dt: float = 1.0) -> Tuple[float, float]:
        """
        Predict the next state (for static targets, position doesn't change).
        
        Args:
            dt: Time step (not used for static model)
        
        Returns:
            Predicted (x, y) position
        """
        # Predict state: x_k|k-1 = F * x_k-1|k-1
        self.x = self.F @ self.x
        
        # Predict covariance: P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return (float(self.x[0, 0]), float(self.x[1, 0]))
    
    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """
        Update the filter with a position measurement.
        
        Args:
            measurement: Observed (x, y) position
        
        Returns:
            Updated (x, y) position estimate
        """
        z = np.array([[measurement[0]], [measurement[1]]], dtype=float)
        
        # Innovation: y = z - H * x_k|k-1
        y = z - self.H @ self.x
        
        # Innovation covariance: S = H * P_k|k-1 * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P_k|k-1 * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x_k|k = x_k|k-1 + K * y
        self.x = self.x + K @ y
        
        # Update covariance: P_k|k = (I - K * H) * P_k|k-1
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P
        
        return (float(self.x[0, 0]), float(self.x[1, 0]))
    
    def get_state(self) -> Tuple[float, float]:
        """Get current position estimate."""
        return (float(self.x[0, 0]), float(self.x[1, 0]))
    
    def get_covariance(self) -> np.ndarray:
        """Get current state covariance matrix."""
        return self.P.copy()


class ConstantVelocityKalmanFilter:
    """
    Kalman filter for tracking targets with constant velocity.
    
    State vector: [x, vx, y, vy] - position and velocity
    Measurement: [x, y] - position measurements only
    
    Assumes constant velocity motion with process noise.
    """
    
    def __init__(self, initial_position: Tuple[float, float],
                 initial_velocity: Optional[Tuple[float, float]] = None,
                 position_uncertainty: float = 1.0,
                 velocity_uncertainty: float = 1.0,
                 process_noise: float = 0.1,
                 measurement_noise: float = 1.0):
        """
        Initialize the constant velocity Kalman filter.
        
        Args:
            initial_position: Initial (x, y) position estimate
            initial_velocity: Initial (vx, vy) velocity estimate (default: 0, 0)
            position_uncertainty: Initial position uncertainty
            velocity_uncertainty: Initial velocity uncertainty
            process_noise: Process noise (acceleration uncertainty)
            measurement_noise: Measurement noise (position measurement uncertainty)
        """
        if initial_velocity is None:
            initial_velocity = (0.0, 0.0)
        
        # State vector: [x, vx, y, vy]
        self.x = np.array([[initial_position[0]], [initial_velocity[0]], 
                          [initial_position[1]], [initial_velocity[1]]], dtype=float)
        
        # State covariance matrix
        self.P = np.diag([position_uncertainty**2, velocity_uncertainty**2,
                         position_uncertainty**2, velocity_uncertainty**2])
        
        # State transition matrix will be set in predict() based on dt
        self.F = np.eye(4)
        
        # Process noise covariance matrix (acceleration noise)
        self.q = process_noise ** 2
        
        # Measurement matrix (observe position only)
        self.H = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0]], dtype=float)
        
        # Measurement noise covariance matrix
        self.R = np.eye(2) * (measurement_noise ** 2)
    
    def _update_transition_matrix(self, dt: float):
        """Update the state transition matrix based on time step."""
        self.F = np.array([[1, dt, 0,  0],
                          [0,  1, 0,  0],
                          [0,  0, 1, dt],
                          [0,  0, 0,  1]], dtype=float)
        
        # Process noise matrix for constant velocity model
        # Assumes acceleration noise
        dt2 = dt * dt
        dt3 = dt2 * dt / 2
        dt4 = dt3 * dt / 2
        
        self.Q = self.q * np.array([[dt4, dt3,   0,   0],
                                   [dt3, dt2,   0,   0],
                                   [  0,   0, dt4, dt3],
                                   [  0,   0, dt3, dt2]], dtype=float)
    
    def predict(self, dt: float = 1.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Predict the next state.
        
        Args:
            dt: Time step
        
        Returns:
            Tuple of (predicted_position, predicted_velocity)
        """
        self._update_transition_matrix(dt)
        
        # Predict state: x_k|k-1 = F * x_k-1|k-1
        self.x = self.F @ self.x
        
        # Predict covariance: P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        position = (float(self.x[0, 0]), float(self.x[2, 0]))
        velocity = (float(self.x[1, 0]), float(self.x[3, 0]))
        return position, velocity
    
    def update(self, measurement: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Update the filter with a position measurement.
        
        Args:
            measurement: Observed (x, y) position
        
        Returns:
            Tuple of (updated_position, updated_velocity)
        """
        z = np.array([[measurement[0]], [measurement[1]]], dtype=float)
        
        # Innovation: y = z - H * x_k|k-1
        y = z - self.H @ self.x
        
        # Innovation covariance: S = H * P_k|k-1 * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P_k|k-1 * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x_k|k = x_k|k-1 + K * y
        self.x = self.x + K @ y
        
        # Update covariance: P_k|k = (I - K * H) * P_k|k-1
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        position = (float(self.x[0, 0]), float(self.x[2, 0]))
        velocity = (float(self.x[1, 0]), float(self.x[3, 0]))
        return position, velocity
    
    def get_state(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get current position and velocity estimates."""
        position = (float(self.x[0, 0]), float(self.x[2, 0]))
        velocity = (float(self.x[1, 0]), float(self.x[3, 0]))
        return position, velocity
    
    def get_covariance(self) -> np.ndarray:
        """Get current state covariance matrix."""
        return self.P.copy()


class ConstantAccelerationKalmanFilter:
    """
    Kalman filter for tracking targets with constant acceleration.
    
    State vector: [x, vx, ax, y, vy, ay] - position, velocity, and acceleration
    Measurement: [x, y] - position measurements only
    
    Assumes constant acceleration motion with process noise.
    """
    
    def __init__(self, initial_position: Tuple[float, float],
                 initial_velocity: Optional[Tuple[float, float]] = None,
                 initial_acceleration: Optional[Tuple[float, float]] = None,
                 position_uncertainty: float = 1.0,
                 velocity_uncertainty: float = 1.0,
                 acceleration_uncertainty: float = 1.0,
                 process_noise: float = 0.1,
                 measurement_noise: float = 1.0):
        """
        Initialize the constant acceleration Kalman filter.
        
        Args:
            initial_position: Initial (x, y) position estimate
            initial_velocity: Initial (vx, vy) velocity estimate (default: 0, 0)
            initial_acceleration: Initial (ax, ay) acceleration estimate (default: 0, 0)
            position_uncertainty: Initial position uncertainty
            velocity_uncertainty: Initial velocity uncertainty
            acceleration_uncertainty: Initial acceleration uncertainty
            process_noise: Process noise (jerk uncertainty)
            measurement_noise: Measurement noise (position measurement uncertainty)
        """
        if initial_velocity is None:
            initial_velocity = (0.0, 0.0)
        if initial_acceleration is None:
            initial_acceleration = (0.0, 0.0)
        
        # State vector: [x, vx, ax, y, vy, ay]
        self.x = np.array([[initial_position[0]], [initial_velocity[0]], [initial_acceleration[0]],
                          [initial_position[1]], [initial_velocity[1]], [initial_acceleration[1]]], dtype=float)
        
        # State covariance matrix
        self.P = np.diag([position_uncertainty**2, velocity_uncertainty**2, acceleration_uncertainty**2,
                         position_uncertainty**2, velocity_uncertainty**2, acceleration_uncertainty**2])
        
        # State transition matrix will be set in predict() based on dt
        self.F = np.eye(6)
        
        # Process noise covariance matrix (jerk noise)
        self.q = process_noise ** 2
        
        # Measurement matrix (observe position only)
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0]], dtype=float)
        
        # Measurement noise covariance matrix
        self.R = np.eye(2) * (measurement_noise ** 2)
    
    def _update_transition_matrix(self, dt: float):
        """Update the state transition matrix based on time step."""
        dt2 = dt * dt / 2
        
        self.F = np.array([[1, dt, dt2, 0,  0,   0],
                          [0,  1,  dt, 0,  0,   0],
                          [0,  0,   1, 0,  0,   0],
                          [0,  0,   0, 1, dt, dt2],
                          [0,  0,   0, 0,  1,  dt],
                          [0,  0,   0, 0,  0,   1]], dtype=float)
        
        # Process noise matrix for constant acceleration model
        # Assumes jerk (derivative of acceleration) noise
        dt3 = dt * dt2 / 3
        dt4 = dt2 * dt2 / 4
        dt5 = dt3 * dt2 / 5
        dt6 = dt3 * dt3 / 6
        
        self.Q = self.q * np.array([[dt6, dt5, dt4,   0,   0,   0],
                                   [dt5, dt4, dt3,   0,   0,   0],
                                   [dt4, dt3, dt2,   0,   0,   0],
                                   [  0,   0,   0, dt6, dt5, dt4],
                                   [  0,   0,   0, dt5, dt4, dt3],
                                   [  0,   0,   0, dt4, dt3, dt2]], dtype=float)
    
    def predict(self, dt: float = 1.0) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Predict the next state.
        
        Args:
            dt: Time step
        
        Returns:
            Tuple of (predicted_position, predicted_velocity, predicted_acceleration)
        """
        self._update_transition_matrix(dt)
        
        # Predict state: x_k|k-1 = F * x_k-1|k-1
        self.x = self.F @ self.x
        
        # Predict covariance: P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        position = (float(self.x[0, 0]), float(self.x[3, 0]))
        velocity = (float(self.x[1, 0]), float(self.x[4, 0]))
        acceleration = (float(self.x[2, 0]), float(self.x[5, 0]))
        return position, velocity, acceleration
    
    def update(self, measurement: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Update the filter with a position measurement.
        
        Args:
            measurement: Observed (x, y) position
        
        Returns:
            Tuple of (updated_position, updated_velocity, updated_acceleration)
        """
        z = np.array([[measurement[0]], [measurement[1]]], dtype=float)
        
        # Innovation: y = z - H * x_k|k-1
        y = z - self.H @ self.x
        
        # Innovation covariance: S = H * P_k|k-1 * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P_k|k-1 * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x_k|k = x_k|k-1 + K * y
        self.x = self.x + K @ y
        
        # Update covariance: P_k|k = (I - K * H) * P_k|k-1
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        
        position = (float(self.x[0, 0]), float(self.x[3, 0]))
        velocity = (float(self.x[1, 0]), float(self.x[4, 0]))
        acceleration = (float(self.x[2, 0]), float(self.x[5, 0]))
        return position, velocity, acceleration
    
    def get_state(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Get current position, velocity, and acceleration estimates."""
        position = (float(self.x[0, 0]), float(self.x[3, 0]))
        velocity = (float(self.x[1, 0]), float(self.x[4, 0]))
        acceleration = (float(self.x[2, 0]), float(self.x[5, 0]))
        return position, velocity, acceleration
    
    def get_covariance(self) -> np.ndarray:
        """Get current state covariance matrix."""
        return self.P.copy()


class TestStaticKalmanFilter(unittest.TestCase):
    """Unit tests for StaticKalmanFilter."""
    
    def setUp(self):
        self.tracker = StaticKalmanFilter(
            initial_position=(0.0, 0.0),
            position_uncertainty=1.0,
            process_noise=0.1,
            measurement_noise=0.5
        )
    
    def test_initialization(self):
        """Test proper initialization."""
        pos = self.tracker.get_state()
        self.assertEqual(pos, (0.0, 0.0))
        
        # Check covariance matrix dimensions
        cov = self.tracker.get_covariance()
        self.assertEqual(cov.shape, (2, 2))
    
    def test_predict_static(self):
        """Test prediction for static model."""
        initial_pos = self.tracker.get_state()
        predicted_pos = self.tracker.predict(dt=1.0)
        
        # For static model, prediction should not change position
        self.assertEqual(initial_pos, predicted_pos)
    
    def test_update_with_measurement(self):
        """Test update with position measurement."""
        measurement = (5.0, 3.0)
        updated_pos = self.tracker.update(measurement)
        
        # Position should move toward measurement
        self.assertNotEqual(updated_pos, (0.0, 0.0))
        self.assertIsInstance(updated_pos[0], float)
        self.assertIsInstance(updated_pos[1], float)
    
    def test_convergence(self):
        """Test convergence to true position with multiple measurements."""
        true_position = (10.0, 5.0)
        measurements = [(10.1, 4.9), (9.9, 5.1), (10.0, 5.0), (9.8, 5.2), (10.2, 4.8)]
        
        for meas in measurements:
            self.tracker.predict()
            self.tracker.update(meas)
        
        final_pos = self.tracker.get_state()
        # Should converge close to true position
        self.assertAlmostEqual(final_pos[0], true_position[0], delta=1.0)
        self.assertAlmostEqual(final_pos[1], true_position[1], delta=1.0)


class TestConstantVelocityKalmanFilter(unittest.TestCase):
    """Unit tests for ConstantVelocityKalmanFilter."""
    
    def setUp(self):
        self.tracker = ConstantVelocityKalmanFilter(
            initial_position=(0.0, 0.0),
            initial_velocity=(1.0, 0.5),
            position_uncertainty=1.0,
            velocity_uncertainty=0.5,
            process_noise=0.1,
            measurement_noise=0.3
        )
    
    def test_initialization(self):
        """Test proper initialization."""
        pos, vel = self.tracker.get_state()
        self.assertEqual(pos, (0.0, 0.0))
        self.assertEqual(vel, (1.0, 0.5))
        
        # Check covariance matrix dimensions
        cov = self.tracker.get_covariance()
        self.assertEqual(cov.shape, (4, 4))
    
    def test_predict_constant_velocity(self):
        """Test prediction for constant velocity model."""
        dt = 2.0
        predicted_pos, predicted_vel = self.tracker.predict(dt)
        
        # Position should change based on velocity
        expected_x = 0.0 + 1.0 * dt  # x = x0 + vx * dt
        expected_y = 0.0 + 0.5 * dt  # y = y0 + vy * dt
        
        self.assertAlmostEqual(predicted_pos[0], expected_x, places=5)
        self.assertAlmostEqual(predicted_pos[1], expected_y, places=5)
        
        # Velocity should remain constant in prediction
        self.assertAlmostEqual(predicted_vel[0], 1.0, places=5)
        self.assertAlmostEqual(predicted_vel[1], 0.5, places=5)
    
    def test_update_with_measurement(self):
        """Test update with position measurement."""
        measurement = (2.0, 1.0)
        updated_pos, updated_vel = self.tracker.update(measurement)
        
        # State should be updated
        self.assertNotEqual(updated_pos, (0.0, 0.0))
        self.assertIsInstance(updated_pos[0], float)
        self.assertIsInstance(updated_pos[1], float)
        self.assertIsInstance(updated_vel[0], float)
        self.assertIsInstance(updated_vel[1], float)
    
    def test_tracking_linear_motion(self):
        """Test tracking object with known linear motion."""
        # Simulate object moving with velocity (2.0, 1.0)
        true_velocity = (2.0, 1.0)
        dt = 1.0
        
        # Set up tracker with correct initial velocity
        tracker = ConstantVelocityKalmanFilter(
            initial_position=(0.0, 0.0),
            initial_velocity=true_velocity,
            measurement_noise=0.1
        )
        
        # Generate perfect measurements (no noise)
        for i in range(5):
            true_pos = (true_velocity[0] * i * dt, true_velocity[1] * i * dt)
            tracker.predict(dt)
            pos, vel = tracker.update(true_pos)
            
            # Should track position accurately
            self.assertAlmostEqual(pos[0], true_pos[0], delta=0.5)
            self.assertAlmostEqual(pos[1], true_pos[1], delta=0.5)


class TestConstantAccelerationKalmanFilter(unittest.TestCase):
    """Unit tests for ConstantAccelerationKalmanFilter."""
    
    def setUp(self):
        self.tracker = ConstantAccelerationKalmanFilter(
            initial_position=(0.0, 0.0),
            initial_velocity=(0.0, 0.0),
            initial_acceleration=(1.0, 0.5),
            position_uncertainty=1.0,
            velocity_uncertainty=0.5,
            acceleration_uncertainty=0.2,
            process_noise=0.05,
            measurement_noise=0.2
        )
    
    def test_initialization(self):
        """Test proper initialization."""
        pos, vel, acc = self.tracker.get_state()
        self.assertEqual(pos, (0.0, 0.0))
        self.assertEqual(vel, (0.0, 0.0))
        self.assertEqual(acc, (1.0, 0.5))
        
        # Check covariance matrix dimensions
        cov = self.tracker.get_covariance()
        self.assertEqual(cov.shape, (6, 6))
    
    def test_predict_constant_acceleration(self):
        """Test prediction for constant acceleration model."""
        dt = 2.0
        predicted_pos, predicted_vel, predicted_acc = self.tracker.predict(dt)
        
        # Position: x = x0 + v0*t + 0.5*a*t²
        expected_x = 0.0 + 0.0 * dt + 0.5 * 1.0 * dt * dt
        expected_y = 0.0 + 0.0 * dt + 0.5 * 0.5 * dt * dt
        
        # Velocity: v = v0 + a*t
        expected_vx = 0.0 + 1.0 * dt
        expected_vy = 0.0 + 0.5 * dt
        
        self.assertAlmostEqual(predicted_pos[0], expected_x, places=5)
        self.assertAlmostEqual(predicted_pos[1], expected_y, places=5)
        self.assertAlmostEqual(predicted_vel[0], expected_vx, places=5)
        self.assertAlmostEqual(predicted_vel[1], expected_vy, places=5)
        
        # Acceleration should remain constant
        self.assertAlmostEqual(predicted_acc[0], 1.0, places=5)
        self.assertAlmostEqual(predicted_acc[1], 0.5, places=5)
    
    def test_update_with_measurement(self):
        """Test update with position measurement."""
        measurement = (2.0, 1.0)
        updated_pos, updated_vel, updated_acc = self.tracker.update(measurement)
        
        # State should be updated
        self.assertNotEqual(updated_pos, (0.0, 0.0))
        self.assertIsInstance(updated_pos[0], float)
        self.assertIsInstance(updated_pos[1], float)
        self.assertIsInstance(updated_vel[0], float)
        self.assertIsInstance(updated_vel[1], float)
        self.assertIsInstance(updated_acc[0], float)
        self.assertIsInstance(updated_acc[1], float)
    
    def test_tracking_accelerated_motion(self):
        """Test tracking object with known constant acceleration."""
        # Simulate object with acceleration (0.5, 0.2)
        true_acceleration = (0.5, 0.2)
        dt = 1.0
        
        # Set up tracker with correct acceleration
        tracker = ConstantAccelerationKalmanFilter(
            initial_position=(0.0, 0.0),
            initial_velocity=(0.0, 0.0),
            initial_acceleration=true_acceleration,
            measurement_noise=0.1
        )
        
        # Generate measurements for accelerated motion
        for i in range(5):
            t = i * dt
            # Position: x = 0.5*a*t²
            true_pos = (0.5 * true_acceleration[0] * t * t, 
                       0.5 * true_acceleration[1] * t * t)
            
            tracker.predict(dt)
            pos, vel, acc = tracker.update(true_pos)
            
            # Should track position accurately
            if i > 0:  # Skip first measurement (t=0, pos=0)
                self.assertAlmostEqual(pos[0], true_pos[0], delta=1.0)
                self.assertAlmostEqual(pos[1], true_pos[1], delta=1.0)
    
    def test_state_estimation_consistency(self):
        """Test that estimated states are consistent with physics."""
        # After prediction, check that position, velocity, acceleration are consistent
        dt = 1.0
        
        # Get initial state
        pos0, vel0, acc0 = self.tracker.get_state()
        
        # Predict
        pos1, vel1, acc1 = self.tracker.predict(dt)
        
        # Check kinematic consistency
        # v = v0 + a*t
        expected_vx = vel0[0] + acc0[0] * dt
        expected_vy = vel0[1] + acc0[1] * dt
        
        self.assertAlmostEqual(vel1[0], expected_vx, places=5)
        self.assertAlmostEqual(vel1[1], expected_vy, places=5)
        
        # x = x0 + v0*t + 0.5*a*t²
        expected_x = pos0[0] + vel0[0] * dt + 0.5 * acc0[0] * dt * dt
        expected_y = pos0[1] + vel0[1] * dt + 0.5 * acc0[1] * dt * dt
        
        self.assertAlmostEqual(pos1[0], expected_x, places=5)
        self.assertAlmostEqual(pos1[1], expected_y, places=5)


if __name__ == '__main__':
    unittest.main()