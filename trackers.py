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


if __name__ == '__main__':
    """Example usage of the Kalman filter trackers."""
    print("Kalman Filter Trackers Example")
    print("=" * 40)
    
    # Example 1: Static tracker
    print("1. Static Tracker Example:")
    static_tracker = StaticKalmanFilter(
        initial_position=(0.0, 0.0),
        position_uncertainty=2.0,
        process_noise=0.1,
        measurement_noise=1.0
    )
    
    # Simulate some noisy measurements of a stationary target at (10, 5)
    true_position = (10.0, 5.0)
    measurements = [(10.2, 4.8), (9.9, 5.1), (10.1, 4.9), (9.8, 5.2)]
    
    print(f"  True position: {true_position}")
    print("  Measurements and estimates:")
    for i, meas in enumerate(measurements):
        static_tracker.predict()
        est_pos = static_tracker.update(meas)
        print(f"    Step {i+1}: measurement={meas}, estimate={est_pos[0]:.2f}, {est_pos[1]:.2f}")
    
    print()
    
    # Example 2: Constant velocity tracker
    print("2. Constant Velocity Tracker Example:")
    cv_tracker = ConstantVelocityKalmanFilter(
        initial_position=(0.0, 0.0),
        initial_velocity=(1.0, 0.5),
        position_uncertainty=1.0,
        velocity_uncertainty=0.5,
        process_noise=0.1,
        measurement_noise=0.5
    )
    
    print("  Tracking object with velocity (1.0, 0.5) m/s:")
    dt = 1.0
    for i in range(5):
        # True position at time t = i * dt
        true_pos = (1.0 * i * dt, 0.5 * i * dt)
        # Add some noise to create measurement
        noise_x = np.random.normal(0, 0.3)
        noise_y = np.random.normal(0, 0.3)
        meas = (true_pos[0] + noise_x, true_pos[1] + noise_y)
        
        cv_tracker.predict(dt)
        est_pos, est_vel = cv_tracker.update(meas)
        print(f"    t={i*dt:.1f}s: true_pos={true_pos}, measurement={meas[0]:.2f}, {meas[1]:.2f}")
        print(f"              estimate_pos={est_pos[0]:.2f}, {est_pos[1]:.2f}, estimate_vel={est_vel[0]:.2f}, {est_vel[1]:.2f}")
    
    print()
    
    # Example 3: Constant acceleration tracker
    print("3. Constant Acceleration Tracker Example:")
    ca_tracker = ConstantAccelerationKalmanFilter(
        initial_position=(0.0, 0.0),
        initial_velocity=(0.0, 0.0),
        initial_acceleration=(0.5, 0.2),
        position_uncertainty=1.0,
        velocity_uncertainty=0.5,
        acceleration_uncertainty=0.2,
        process_noise=0.05,
        measurement_noise=0.3
    )
    
    print("  Tracking object with acceleration (0.5, 0.2) m/s²:")
    dt = 1.0
    for i in range(5):
        t = i * dt
        # True position: x = 0.5*a*t² = 0.25*t², y = 0.1*t²
        true_pos = (0.25 * t * t, 0.1 * t * t)
        # Add noise
        noise_x = np.random.normal(0, 0.2)
        noise_y = np.random.normal(0, 0.2)
        meas = (true_pos[0] + noise_x, true_pos[1] + noise_y)
        
        ca_tracker.predict(dt)
        est_pos, est_vel, est_acc = ca_tracker.update(meas)
        print(f"    t={t:.1f}s: true_pos={true_pos[0]:.2f}, {true_pos[1]:.2f}, measurement={meas[0]:.2f}, {meas[1]:.2f}")
        print(f"             estimate_pos={est_pos[0]:.2f}, {est_pos[1]:.2f}, vel={est_vel[0]:.2f}, {est_vel[1]:.2f}, acc={est_acc[0]:.2f}, {est_acc[1]:.2f}")
    
    print("\nKalman filter examples completed!")