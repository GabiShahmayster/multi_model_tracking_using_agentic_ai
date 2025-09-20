#!/usr/bin/env python3
"""
Wrong Model Tracking Demonstration

This script demonstrates the effects of using an incorrect motion model for tracking.
The target follows a specific motion profile:
- 0-50s: Static (no motion)
- 50-100s: Constant velocity (10 m/s in x-direction)

However, the tracker uses a static motion model throughout the entire simulation,
which is correct for the first 50 seconds but completely wrong for the second 50 seconds.

This shows how model mismatch affects tracking performance, particularly:
- Position tracking errors during the constant velocity phase
- Velocity estimation errors (static model assumes zero velocity)
"""

import numpy as np
import matplotlib.pyplot as plt
from motion import MotionSimulator
from trackers import StaticKalmanFilter


def main():
    print("Wrong Model Tracking Demonstration")
    print("=" * 50)

    # Simulation parameters
    duration = 100.0
    dt = 0.5  # Time step
    measurement_noise_std = 0.5  # Standard deviation of measurement noise

    print(f"Simulation duration: {duration} seconds")
    print(f"Time step: {dt} seconds")
    print(f"Measurement noise: {measurement_noise_std} m std dev")
    print()

    # Motion profile
    print("Target motion profile:")
    print("  0-50s:  Static (no motion)")
    print(" 50-100s: Constant velocity (10 m/s in x-direction)")
    print()
    print("Tracker model: Static motion model (WRONG for 50-100s)")
    print()

    # Create motion simulator
    sim = MotionSimulator(x0=0.0, y0=0.0, vx0=0.0, vy0=0.0)
    sim.add_segment(sim.STATIC, 50.0)  # Static for 50 seconds
    sim.add_segment(sim.CONSTANT_VELOCITY, 50.0)  # Constant velocity for 50 seconds

    # Set the constant velocity for the second segment
    # The velocity will be 10 m/s in x-direction, 0 in y-direction
    # This happens automatically since the simulator maintains velocity from previous segment
    # We need to manually set this by getting the end state and creating a new segment

    # Actually, let's create this more explicitly
    sim = MotionSimulator(x0=0.0, y0=0.0, vx0=0.0, vy0=0.0)
    sim.add_segment(sim.STATIC, 50.0)  # Static for 50 seconds

    # For constant velocity phase, we need to set the velocity
    # Since the target was static, we need to give it an instantaneous velocity change
    # We'll simulate this by adding a very short acceleration phase followed by constant velocity
    sim.add_segment(sim.CONSTANT_ACCELERATION, 0.1, ax=100.0, ay=0.0)  # Quick acceleration to 10 m/s
    sim.add_segment(sim.CONSTANT_VELOCITY, 49.9)  # Constant velocity for remaining time

    print(f"Motion simulator created with {len(sim.segments)} segments")
    print(f"Total simulation duration: {sim.duration} seconds")
    print()

    # Initialize static Kalman filter (wrong model for second half)
    initial_position = (0.0, 0.0)
    tracker = StaticKalmanFilter(
        initial_position=initial_position,
        position_uncertainty=1.0,
        process_noise=0.1,  # Small process noise for static model
        measurement_noise=measurement_noise_std
    )

    print("Static Kalman filter initialized")
    print(f"  Initial position: {initial_position}")
    print(f"  Position uncertainty: 1.0 m")
    print(f"  Process noise: 0.1 m")
    print(f"  Measurement noise: {measurement_noise_std} m")
    print()

    # Generate time steps
    times = np.arange(0, duration + dt, dt)
    n_steps = len(times)

    print(f"Generating {n_steps} time steps for simulation...")

    # Arrays to store simulation data
    true_positions = np.zeros((n_steps, 2))
    true_velocities = np.zeros((n_steps, 2))
    measurements = np.zeros((n_steps, 2))
    estimated_positions = np.zeros((n_steps, 2))
    estimated_velocities = np.zeros((n_steps, 2))  # Will be zero for static model
    position_errors = np.zeros((n_steps, 2))
    velocity_errors = np.zeros((n_steps, 2))
    innovations = np.zeros((n_steps, 2))  # Innovation sequence (measurement - prediction)

    # Generate true motion and noisy measurements
    np.random.seed(42)  # For reproducible results
    for i, t in enumerate(times):
        # Get true position and velocity
        true_pos = sim.get_position(t)
        true_vel = sim.get_velocity(t)

        true_positions[i] = true_pos
        true_velocities[i] = true_vel

        # Generate noisy measurement
        noise = np.random.normal(0, measurement_noise_std, 2)
        measurements[i] = np.array(true_pos) + noise

    print("True motion and measurements generated")
    print()

    # Run tracking simulation
    print("Running tracking simulation...")
    for i, t in enumerate(times):
        # Get prediction before update for innovation calculation
        if i == 0:
            predicted_pos = tracker.get_state()
        else:
            tracker.predict()
            predicted_pos = tracker.get_state()

        # Calculate innovation (measurement - prediction)
        measurement = measurements[i]
        innovations[i] = np.array(measurement) - np.array(predicted_pos)

        # Update tracker with measurement
        tracker.update(measurement)

        # Get tracker estimates
        estimated_pos = tracker.get_state()
        estimated_vel = (0.0, 0.0)  # Static model always estimates zero velocity

        estimated_positions[i] = estimated_pos
        estimated_velocities[i] = estimated_vel

        # Calculate errors
        position_errors[i] = np.array(true_positions[i]) - np.array(estimated_pos)
        velocity_errors[i] = np.array(true_velocities[i]) - np.array(estimated_vel)

    print("Tracking simulation completed")
    print()

    # Calculate error statistics
    print("Calculating error statistics...")

    # Position error magnitudes
    position_error_mag = np.sqrt(position_errors[:, 0]**2 + position_errors[:, 1]**2)
    velocity_error_mag = np.sqrt(velocity_errors[:, 0]**2 + velocity_errors[:, 1]**2)
    innovation_mag = np.sqrt(innovations[:, 0]**2 + innovations[:, 1]**2)

    # Statistics for first 50 seconds (static phase - correct model)
    static_indices = times <= 50.0
    static_pos_rmse = np.sqrt(np.mean(position_error_mag[static_indices]**2))
    static_vel_rmse = np.sqrt(np.mean(velocity_error_mag[static_indices]**2))

    # Statistics for last 50 seconds (motion phase - wrong model)
    motion_indices = times > 50.0
    motion_pos_rmse = np.sqrt(np.mean(position_error_mag[motion_indices]**2))
    motion_vel_rmse = np.sqrt(np.mean(velocity_error_mag[motion_indices]**2))

    # Overall statistics
    overall_pos_rmse = np.sqrt(np.mean(position_error_mag**2))
    overall_vel_rmse = np.sqrt(np.mean(velocity_error_mag**2))

    print("Error Statistics:")
    print(f"  Static phase (0-50s) - Position RMSE: {static_pos_rmse:.3f} m")
    print(f"  Static phase (0-50s) - Velocity RMSE: {static_vel_rmse:.3f} m/s")
    print(f"  Motion phase (50-100s) - Position RMSE: {motion_pos_rmse:.3f} m")
    print(f"  Motion phase (50-100s) - Velocity RMSE: {motion_vel_rmse:.3f} m/s")
    print(f"  Overall - Position RMSE: {overall_pos_rmse:.3f} m")
    print(f"  Overall - Velocity RMSE: {overall_vel_rmse:.3f} m/s")
    print()

    # Create comprehensive plots
    print("Creating plots...")

    plt.style.use('default')
    fig = plt.figure(figsize=(16, 14))  # Increased height for additional plot

    # Plot 1: True vs Estimated Positions
    ax1 = plt.subplot(4, 2, 1)
    plt.plot(times, true_positions[:, 0], 'b-', linewidth=2, label='True X Position')
    plt.plot(times, estimated_positions[:, 0], 'r--', linewidth=2, label='Estimated X Position')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.title('X Position: True vs Estimated')
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax2 = plt.subplot(4, 2, 2)
    plt.plot(times, true_positions[:, 1], 'b-', linewidth=2, label='True Y Position')
    plt.plot(times, estimated_positions[:, 1], 'r--', linewidth=2, label='Estimated Y Position')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (m)')
    plt.title('Y Position: True vs Estimated')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: True vs Estimated Velocities
    ax3 = plt.subplot(4, 2, 3)
    plt.plot(times, true_velocities[:, 0], 'b-', linewidth=2, label='True X Velocity')
    plt.plot(times, estimated_velocities[:, 0], 'r--', linewidth=2, label='Estimated X Velocity')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.xlabel('Time (s)')
    plt.ylabel('X Velocity (m/s)')
    plt.title('X Velocity: True vs Estimated')
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax4 = plt.subplot(4, 2, 4)
    plt.plot(times, true_velocities[:, 1], 'b-', linewidth=2, label='True Y Velocity')
    plt.plot(times, estimated_velocities[:, 1], 'r--', linewidth=2, label='Estimated Y Velocity')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Velocity (m/s)')
    plt.title('Y Velocity: True vs Estimated')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 5: Position Tracking Errors
    ax5 = plt.subplot(4, 2, 5)
    plt.plot(times, position_errors[:, 0], 'r-', linewidth=1.5, label='X Position Error')
    plt.plot(times, position_errors[:, 1], 'g-', linewidth=1.5, label='Y Position Error')
    plt.plot(times, position_error_mag, 'k-', linewidth=2, label='Position Error Magnitude')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error (m)')
    plt.title('Position Tracking Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 6: Velocity Tracking Errors
    ax6 = plt.subplot(4, 2, 6)
    plt.plot(times, velocity_errors[:, 0], 'r-', linewidth=1.5, label='X Velocity Error')
    plt.plot(times, velocity_errors[:, 1], 'g-', linewidth=1.5, label='Y Velocity Error')
    plt.plot(times, velocity_error_mag, 'k-', linewidth=2, label='Velocity Error Magnitude')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity Error (m/s)')
    plt.title('Velocity Tracking Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 7: Innovation Sequence X
    ax7 = plt.subplot(4, 2, 7)
    plt.plot(times, innovations[:, 0], 'b-', linewidth=1.5, label='X Innovation')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Innovation (m)')
    plt.title('Innovation Sequence X (Measurement - Prediction)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 8: Innovation Sequence Magnitude
    ax8 = plt.subplot(4, 2, 8)
    plt.plot(times, innovation_mag, 'r-', linewidth=2, label='Innovation Magnitude')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Innovation Magnitude (m)')
    plt.title('Innovation Sequence Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle('Wrong Model Tracking: Static Filter vs Moving Target', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save the plot
    plot_filename = 'wrong_model_tracking_errors.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Comprehensive plots saved to: {plot_filename}")

    print()
    print("=" * 50)
    print("SIMULATION SUMMARY")
    print("=" * 50)
    print("This simulation demonstrates the effects of using a wrong motion model:")
    print("- Target: Static (0-50s) → Constant velocity 10 m/s (50-100s)")
    print("- Tracker: Static model throughout (correct only for first 50s)")
    print()
    print("Key observations:")
    print(f"- During static phase: Good tracking (Pos RMSE = {static_pos_rmse:.3f} m)")
    print(f"- During motion phase: Poor tracking (Pos RMSE = {motion_pos_rmse:.3f} m)")
    print(f"- Velocity estimation: Always wrong during motion (RMSE = {motion_vel_rmse:.3f} m/s)")
    print("- Static model cannot adapt to the target's motion")


if __name__ == '__main__':
    main()