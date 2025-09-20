#!/usr/bin/env python3
"""
Correct Model Tracking Demonstration

This script demonstrates proper tracking using an IMM (Interacting Multiple Model) filter
for the same target motion profile as wrong_model.py:
- 0-50s: Static (no motion)
- 50-100s: Constant velocity (10 m/s in x-direction)

The tracker uses an IMM filter with two models:
- Model 0: Static motion model
- Model 1: Constant velocity motion model

The IMM algorithm automatically switches between models based on which one
best explains the current measurements, providing optimal tracking performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add base directory to path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'base'))

from motion import MotionSimulator
from imm import TwoModelIMM


def main():
    print("Correct Model Tracking Demonstration (IMM Filter)")
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
    print("Tracker model: IMM Filter (Static + Constant Velocity models)")
    print()

    # Create motion simulator (same as wrong_model.py)
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

    # Initialize IMM tracker with static and constant velocity models
    initial_position = (0.0, 0.0)
    initial_velocity = (0.0, 0.0)

    # Model transition matrix - 2 models: Static, CV
    model_transition_matrix = np.array([
        [0.95, 0.05],  # Static -> [Static, CV]
        [0.05, 0.95]   # CV -> [Static, CV]
    ])

    # Initial model probabilities (start assuming static)
    initial_model_probabilities = np.array([0.9, 0.1])  # [Static, CV]

    tracker = TwoModelIMM(
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        model_transition_matrix=model_transition_matrix,
        initial_model_probabilities=initial_model_probabilities,
        position_uncertainty=1.0,
        velocity_uncertainty=1.0,
        process_noise_static=0.1,
        process_noise_cv=0.1,
        measurement_noise=measurement_noise_std
    )

    print("IMM tracker initialized")
    print(f"  Initial position: {initial_position}")
    print(f"  Initial velocity: {initial_velocity}")
    print(f"  Models: Static + Constant Velocity (2-model IMM)")
    print(f"  Initial model probs: {initial_model_probabilities}")
    print(f"  Model transition matrix:")
    print(f"    {model_transition_matrix}")
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
    estimated_velocities = np.zeros((n_steps, 2))
    position_errors = np.zeros((n_steps, 2))
    velocity_errors = np.zeros((n_steps, 2))
    model_probabilities = np.zeros((n_steps, 2))  # [Static, CV] probabilities
    innovations_static = np.zeros((n_steps, 2))  # Innovations for static model
    innovations_cv = np.zeros((n_steps, 2))  # Innovations for CV model

    # Generate true motion and noisy measurements (same as wrong_model.py)
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
    print("Running IMM tracking simulation...")
    for i, t in enumerate(times):
        # Update tracker with measurement
        measurement = measurements[i]
        tracker.update(measurement)

        # Get tracker estimates
        estimated_state = tracker.get_state()
        estimated_pos = estimated_state[0]  # Position
        estimated_vel = estimated_state[1]  # Velocity
        model_probs = tracker.get_model_probabilities()
        innovations = tracker.get_innovations()

        estimated_positions[i] = estimated_pos
        estimated_velocities[i] = estimated_vel
        model_probabilities[i] = model_probs
        innovations_static[i] = innovations[0]  # Static model innovations
        innovations_cv[i] = innovations[1]  # CV model innovations

        # Calculate errors
        position_errors[i] = np.array(true_positions[i]) - np.array(estimated_pos)
        velocity_errors[i] = np.array(true_velocities[i]) - np.array(estimated_vel)

        # Predict for next time step
        if i < n_steps - 1:
            tracker.predict(dt)

    print("IMM tracking simulation completed")
    print()

    # Calculate error statistics
    print("Calculating error statistics...")

    # Position error magnitudes
    position_error_mag = np.sqrt(position_errors[:, 0]**2 + position_errors[:, 1]**2)
    velocity_error_mag = np.sqrt(velocity_errors[:, 0]**2 + velocity_errors[:, 1]**2)
    innovation_static_mag = np.sqrt(innovations_static[:, 0]**2 + innovations_static[:, 1]**2)
    innovation_cv_mag = np.sqrt(innovations_cv[:, 0]**2 + innovations_cv[:, 1]**2)

    # Statistics for first 50 seconds (static phase)
    static_indices = times <= 50.0
    static_pos_rmse = np.sqrt(np.mean(position_error_mag[static_indices]**2))
    static_vel_rmse = np.sqrt(np.mean(velocity_error_mag[static_indices]**2))

    # Statistics for last 50 seconds (motion phase)
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

    # Model probability statistics
    print("Model Probability Statistics:")
    static_prob_avg = np.mean(model_probabilities[static_indices, 0])
    motion_cv_prob_avg = np.mean(model_probabilities[motion_indices, 1])
    print(f"  Static phase - Average Static model probability: {static_prob_avg:.3f}")
    print(f"  Motion phase - Average CV model probability: {motion_cv_prob_avg:.3f}")
    print()

    # Create comprehensive plots including model probabilities
    print("Creating plots...")

    plt.style.use('default')
    fig = plt.figure(figsize=(18, 20))  # Increased height for additional plots

    # Plot 1: True vs Estimated Positions
    ax1 = plt.subplot(5, 2, 1)
    plt.plot(times, true_positions[:, 0], 'b-', linewidth=2, label='True X Position')
    plt.plot(times, estimated_positions[:, 0], 'r--', linewidth=2, label='Estimated X Position')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.title('X Position: True vs Estimated (IMM)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax2 = plt.subplot(5, 2, 2)
    plt.plot(times, true_positions[:, 1], 'b-', linewidth=2, label='True Y Position')
    plt.plot(times, estimated_positions[:, 1], 'r--', linewidth=2, label='Estimated Y Position')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (m)')
    plt.title('Y Position: True vs Estimated (IMM)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: True vs Estimated Velocities
    ax3 = plt.subplot(5, 2, 3)
    plt.plot(times, true_velocities[:, 0], 'b-', linewidth=2, label='True X Velocity')
    plt.plot(times, estimated_velocities[:, 0], 'r--', linewidth=2, label='Estimated X Velocity')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.xlabel('Time (s)')
    plt.ylabel('X Velocity (m/s)')
    plt.title('X Velocity: True vs Estimated (IMM)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax4 = plt.subplot(5, 2, 4)
    plt.plot(times, true_velocities[:, 1], 'b-', linewidth=2, label='True Y Velocity')
    plt.plot(times, estimated_velocities[:, 1], 'r--', linewidth=2, label='Estimated Y Velocity')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Velocity (m/s)')
    plt.title('Y Velocity: True vs Estimated (IMM)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 5: Position Tracking Errors
    ax5 = plt.subplot(5, 2, 5)
    plt.plot(times, position_errors[:, 0], 'r-', linewidth=1.5, label='X Position Error')
    plt.plot(times, position_errors[:, 1], 'g-', linewidth=1.5, label='Y Position Error')
    plt.plot(times, position_error_mag, 'k-', linewidth=2, label='Position Error Magnitude')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error (m)')
    plt.title('Position Tracking Errors (IMM)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 6: Velocity Tracking Errors
    ax6 = plt.subplot(5, 2, 6)
    plt.plot(times, velocity_errors[:, 0], 'r-', linewidth=1.5, label='X Velocity Error')
    plt.plot(times, velocity_errors[:, 1], 'g-', linewidth=1.5, label='Y Velocity Error')
    plt.plot(times, velocity_error_mag, 'k-', linewidth=2, label='Velocity Error Magnitude')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity Error (m/s)')
    plt.title('Velocity Tracking Errors (IMM)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 7: Model Probabilities
    ax7 = plt.subplot(5, 2, 7)
    plt.plot(times, model_probabilities[:, 0], 'b-', linewidth=2, label='Static Model Prob')
    plt.plot(times, model_probabilities[:, 1], 'r-', linewidth=2, label='CV Model Prob')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='50% threshold')
    plt.xlabel('Time (s)')
    plt.ylabel('Model Probability')
    plt.title('2-Model IMM Probabilities (Static + CV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    # Plot 8: Innovation Sequences for Static Model
    ax8 = plt.subplot(5, 2, 8)
    plt.plot(times, innovation_static_mag, 'b-', linewidth=2, label='Static Model Innovation')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Innovation Magnitude (m)')
    plt.title('Static Model Innovation Sequence')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 9: Innovation Sequences for CV Model
    ax9 = plt.subplot(5, 2, 9)
    plt.plot(times, innovation_cv_mag, 'r-', linewidth=2, label='CV Model Innovation')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Innovation Magnitude (m)')
    plt.title('CV Model Innovation Sequence')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 10: Innovation Comparison
    ax10 = plt.subplot(5, 2, 10)
    plt.plot(times, innovation_static_mag, 'b-', linewidth=2, label='Static Model')
    plt.plot(times, innovation_cv_mag, 'r-', linewidth=2, label='CV Model')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Innovation Magnitude (m)')
    plt.title('Innovation Comparison: Static vs CV')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle('Correct Model Tracking: 2-Model IMM Filter (Static + CV)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save the plot
    os.makedirs('../results/tracking_imm_2_model', exist_ok=True)
    plot_filename = '../results/tracking_imm_2_model/correct_model_tracking_performance.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Comprehensive plots saved to: {plot_filename}")

    print()
    print("=" * 50)
    print("SIMULATION SUMMARY")
    print("=" * 50)
    print("This simulation demonstrates proper tracking using a 2-model IMM filter:")
    print("- Target: Static (0-50s) → Constant velocity 10 m/s (50-100s)")
    print("- Tracker: 2-Model IMM with Static + Constant Velocity models only")
    print()
    print("Key observations:")
    print(f"- During static phase: Good tracking (Pos RMSE = {static_pos_rmse:.3f} m)")
    print(f"- During motion phase: Good tracking (Pos RMSE = {motion_pos_rmse:.3f} m)")
    print(f"- Velocity estimation: Accurate during motion (RMSE = {motion_vel_rmse:.3f} m/s)")
    print(f"- Model switching: Static prob = {static_prob_avg:.3f} → CV prob = {motion_cv_prob_avg:.3f}")
    print("- 2-Model IMM adapts automatically to changing target dynamics")


if __name__ == '__main__':
    main()