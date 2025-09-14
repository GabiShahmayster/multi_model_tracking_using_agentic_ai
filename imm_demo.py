#!/usr/bin/env python3
"""
IMM Tracker Demonstration Script

Demonstrates the capabilities of the IMM (Interacting Multiple Model) tracker
using the same motion profile as main.py:
- 0-10s: Static (no motion)
- 10-20s: Constant acceleration (1 m/s² in x-direction)
- 20-100s: Constant velocity (maintaining velocity from acceleration phase)

The script generates comprehensive plots showing:
1. Target position, velocity, acceleration (per axis)
2. IMM estimates vs truth
3. Model probabilities over time
4. Tracking errors
5. Model likelihood evolution

This demonstrates how the IMM algorithm adapts to changing motion dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from motion import MotionSimulator
from imm import IMMTracker


def main():
    print("IMM Tracker Demonstration")
    print("=" * 50)
    
    # Simulation parameters
    duration = 100.0
    dt = 0.5  # Time step for high resolution
    measurement_noise_std = 0.3  # Standard deviation of measurement noise
    
    print(f"Simulation duration: {duration} seconds")
    print(f"Time step: {dt} seconds")
    print(f"Measurement noise: {measurement_noise_std} m std dev")
    print()
    
    # Motion profile (same as main.py)
    print("Motion profile:")
    print("  0-10s:  Static (no motion)")
    print(" 10-20s:  Constant acceleration (1 m/s² in x-direction)")
    print(" 20-100s: Constant velocity")
    print()
    
    # Create motion simulator
    sim = MotionSimulator(x0=0.0, y0=0.0, vx0=0.0, vy0=0.0)
    sim.add_segment(sim.STATIC, 10.0)  # Static for 10 seconds
    sim.add_segment(sim.CONSTANT_ACCELERATION, 10.0, ax=1.0, ay=0.0)  # 1 m/s² x-acceleration for 10 seconds
    sim.add_segment(sim.CONSTANT_VELOCITY, 80.0)  # Constant velocity for 80 seconds
    
    # Create IMM tracker
    # Use moderate transition probabilities to allow model switching
    transition_matrix = np.array([
        [0.90, 0.05, 0.05],  # From static: likely to stay static
        [0.10, 0.85, 0.05],  # From CV: likely to stay CV or go to static
        [0.05, 0.15, 0.80]   # From CA: likely to stay CA or switch to CV
    ])
    
    imm = IMMTracker(
        initial_position=(0.0, 0.0),
        initial_velocity=(0.0, 0.0),
        initial_acceleration=(0.0, 0.0),
        model_transition_matrix=transition_matrix,
        initial_model_probabilities=np.array([0.7, 0.2, 0.1]),  # Start assuming static
        measurement_noise=measurement_noise_std
    )
    
    # Data collection arrays
    times = []
    true_positions = []
    true_velocities = []
    true_accelerations = []
    measurements = []
    imm_positions = []
    imm_velocities = []
    imm_accelerations = []
    model_probabilities = []
    
    print("Running simulation...")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Simulation loop
    t = 0.0
    while t <= duration:
        # Get true state
        true_pos = sim.get_position(t)
        true_vel = sim.get_velocity(t)
        
        # Calculate true acceleration based on motion phase
        if t < 10.0:
            true_acc = (0.0, 0.0)  # Static phase
        elif t < 20.0:
            true_acc = (1.0, 0.0)  # Acceleration phase
        else:
            true_acc = (0.0, 0.0)  # Constant velocity phase
        
        # Generate noisy measurement
        noise_x = np.random.normal(0, measurement_noise_std)
        noise_y = np.random.normal(0, measurement_noise_std)
        measurement = (true_pos[0] + noise_x, true_pos[1] + noise_y)
        
        # IMM predict and update
        if t > 0:  # Skip prediction on first iteration
            imm.predict(dt)
        imm_pos = imm.update(measurement)
        imm_pos_full, imm_vel, imm_acc = imm.get_state()
        model_probs = imm.get_model_probabilities()
        
        # Store data
        times.append(t)
        true_positions.append(true_pos)
        true_velocities.append(true_vel)
        true_accelerations.append(true_acc)
        measurements.append(measurement)
        imm_positions.append(imm_pos)
        imm_velocities.append(imm_vel)
        imm_accelerations.append(imm_acc)
        model_probabilities.append(model_probs.copy())
        
        t += dt
    
    # Convert to numpy arrays for easier processing
    times = np.array(times)
    true_positions = np.array(true_positions)
    true_velocities = np.array(true_velocities)
    true_accelerations = np.array(true_accelerations)
    measurements = np.array(measurements)
    imm_positions = np.array(imm_positions)
    imm_velocities = np.array(imm_velocities)
    imm_accelerations = np.array(imm_accelerations)
    model_probabilities = np.array(model_probabilities)
    
    print(f"Collected {len(times)} data points")
    print()
    
    # Calculate tracking errors
    pos_errors = imm_positions - true_positions
    vel_errors = imm_velocities - true_velocities
    acc_errors = imm_accelerations - true_accelerations
    
    # Create comprehensive plots
    create_state_plots(times, true_positions, true_velocities, true_accelerations,
                      imm_positions, imm_velocities, imm_accelerations, measurements)
    
    create_model_probability_plots(times, model_probabilities)
    
    create_error_plots(times, pos_errors, vel_errors, acc_errors)
    
    # Print summary statistics
    print_summary_statistics(pos_errors, vel_errors, acc_errors)
    
    print("All plots saved successfully!")
    print("IMM demonstration completed.")


def create_state_plots(times, true_pos, true_vel, true_acc, imm_pos, imm_vel, imm_acc, measurements):
    """Create plots for position, velocity, and acceleration."""
    print("Creating state plots...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('IMM Tracker: Target State Estimation', fontsize=16, fontweight='bold')
    
    # Position plots
    axes[0, 0].plot(times, true_pos[:, 0], 'b-', linewidth=2, label='True X Position')
    axes[0, 0].plot(times, imm_pos[:, 0], 'r--', linewidth=2, label='IMM Estimate')
    axes[0, 0].scatter(times[::4], measurements[::4, 0], c='gray', s=10, alpha=0.5, label='Measurements')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('X Position (m)')
    axes[0, 0].set_title('X Position vs Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=10, color='orange', linestyle=':', alpha=0.7, label='Phase transitions')
    axes[0, 0].axvline(x=20, color='orange', linestyle=':', alpha=0.7)
    
    axes[0, 1].plot(times, true_pos[:, 1], 'b-', linewidth=2, label='True Y Position')
    axes[0, 1].plot(times, imm_pos[:, 1], 'r--', linewidth=2, label='IMM Estimate')
    axes[0, 1].scatter(times[::4], measurements[::4, 1], c='gray', s=10, alpha=0.5, label='Measurements')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Y Position (m)')
    axes[0, 1].set_title('Y Position vs Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=10, color='orange', linestyle=':', alpha=0.7)
    axes[0, 1].axvline(x=20, color='orange', linestyle=':', alpha=0.7)
    
    # Velocity plots
    axes[1, 0].plot(times, true_vel[:, 0], 'b-', linewidth=2, label='True X Velocity')
    axes[1, 0].plot(times, imm_vel[:, 0], 'r--', linewidth=2, label='IMM Estimate')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('X Velocity (m/s)')
    axes[1, 0].set_title('X Velocity vs Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=10, color='orange', linestyle=':', alpha=0.7)
    axes[1, 0].axvline(x=20, color='orange', linestyle=':', alpha=0.7)
    
    axes[1, 1].plot(times, true_vel[:, 1], 'b-', linewidth=2, label='True Y Velocity')
    axes[1, 1].plot(times, imm_vel[:, 1], 'r--', linewidth=2, label='IMM Estimate')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Y Velocity (m/s)')
    axes[1, 1].set_title('Y Velocity vs Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=10, color='orange', linestyle=':', alpha=0.7)
    axes[1, 1].axvline(x=20, color='orange', linestyle=':', alpha=0.7)
    
    # Acceleration plots
    axes[2, 0].plot(times, true_acc[:, 0], 'b-', linewidth=2, label='True X Acceleration')
    axes[2, 0].plot(times, imm_acc[:, 0], 'r--', linewidth=2, label='IMM Estimate')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('X Acceleration (m/s²)')
    axes[2, 0].set_title('X Acceleration vs Time')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].axvline(x=10, color='orange', linestyle=':', alpha=0.7)
    axes[2, 0].axvline(x=20, color='orange', linestyle=':', alpha=0.7)
    
    axes[2, 1].plot(times, true_acc[:, 1], 'b-', linewidth=2, label='True Y Acceleration')
    axes[2, 1].plot(times, imm_acc[:, 1], 'r--', linewidth=2, label='IMM Estimate')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Y Acceleration (m/s²)')
    axes[2, 1].set_title('Y Acceleration vs Time')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axvline(x=10, color='orange', linestyle=':', alpha=0.7)
    axes[2, 1].axvline(x=20, color='orange', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('imm_state_estimation.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_model_probability_plots(times, model_probs):
    """Create plots showing model probabilities over time."""
    print("Creating model probability plots...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('IMM Model Probabilities and Selection', fontsize=16, fontweight='bold')
    
    # Model probabilities over time
    ax1.plot(times, model_probs[:, 0], 'g-', linewidth=2, label='Static Model')
    ax1.plot(times, model_probs[:, 1], 'b-', linewidth=2, label='Constant Velocity Model')
    ax1.plot(times, model_probs[:, 2], 'r-', linewidth=2, label='Constant Acceleration Model')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Model Probability')
    ax1.set_title('Model Probabilities vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    ax1.axvline(x=10, color='orange', linestyle=':', alpha=0.7, label='Phase transitions')
    ax1.axvline(x=20, color='orange', linestyle=':', alpha=0.7)
    
    # Most likely model over time
    most_likely_models = np.argmax(model_probs, axis=1)
    model_names = ['Static', 'Const. Velocity', 'Const. Acceleration']
    colors = ['green', 'blue', 'red']
    
    for i in range(len(most_likely_models)):
        ax2.scatter(times[i], most_likely_models[i], c=colors[most_likely_models[i]], 
                   s=20, alpha=0.7)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Most Likely Model')
    ax2.set_title('Most Likely Model vs Time')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(model_names)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=10, color='orange', linestyle=':', alpha=0.7)
    ax2.axvline(x=20, color='orange', linestyle=':', alpha=0.7)
    
    # Add phase annotations
    ax1.text(5, 0.9, 'Static Phase', ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax1.text(15, 0.9, 'Acceleration Phase', ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax1.text(60, 0.9, 'Constant Velocity Phase', ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig('imm_model_probabilities.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_error_plots(times, pos_errors, vel_errors, acc_errors):
    """Create plots showing tracking errors."""
    print("Creating error analysis plots...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('IMM Tracker: Estimation Errors', fontsize=16, fontweight='bold')
    
    # Position errors
    axes[0, 0].plot(times, pos_errors[:, 0], 'r-', linewidth=1, alpha=0.8)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('X Position Error (m)')
    axes[0, 0].set_title('X Position Estimation Error')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 0].axvline(x=10, color='orange', linestyle=':', alpha=0.7)
    axes[0, 0].axvline(x=20, color='orange', linestyle=':', alpha=0.7)
    
    axes[0, 1].plot(times, pos_errors[:, 1], 'r-', linewidth=1, alpha=0.8)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Y Position Error (m)')
    axes[0, 1].set_title('Y Position Estimation Error')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 1].axvline(x=10, color='orange', linestyle=':', alpha=0.7)
    axes[0, 1].axvline(x=20, color='orange', linestyle=':', alpha=0.7)
    
    # Velocity errors
    axes[1, 0].plot(times, vel_errors[:, 0], 'b-', linewidth=1, alpha=0.8)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('X Velocity Error (m/s)')
    axes[1, 0].set_title('X Velocity Estimation Error')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].axvline(x=10, color='orange', linestyle=':', alpha=0.7)
    axes[1, 0].axvline(x=20, color='orange', linestyle=':', alpha=0.7)
    
    axes[1, 1].plot(times, vel_errors[:, 1], 'b-', linewidth=1, alpha=0.8)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Y Velocity Error (m/s)')
    axes[1, 1].set_title('Y Velocity Estimation Error')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 1].axvline(x=10, color='orange', linestyle=':', alpha=0.7)
    axes[1, 1].axvline(x=20, color='orange', linestyle=':', alpha=0.7)
    
    # Acceleration errors
    axes[2, 0].plot(times, acc_errors[:, 0], 'g-', linewidth=1, alpha=0.8)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('X Acceleration Error (m/s²)')
    axes[2, 0].set_title('X Acceleration Estimation Error')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[2, 0].axvline(x=10, color='orange', linestyle=':', alpha=0.7)
    axes[2, 0].axvline(x=20, color='orange', linestyle=':', alpha=0.7)
    
    axes[2, 1].plot(times, acc_errors[:, 1], 'g-', linewidth=1, alpha=0.8)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Y Acceleration Error (m/s²)')
    axes[2, 1].set_title('Y Acceleration Estimation Error')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[2, 1].axvline(x=10, color='orange', linestyle=':', alpha=0.7)
    axes[2, 1].axvline(x=20, color='orange', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('imm_tracking_errors.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create RMS error plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate RMS errors in sliding windows
    window_size = 20  # 10 second windows at 0.5s intervals
    window_times = []
    pos_rms = []
    vel_rms = []
    
    for i in range(window_size, len(times), window_size//2):
        start_idx = i - window_size
        end_idx = i
        
        window_times.append(times[i])
        
        pos_rms_x = np.sqrt(np.mean(pos_errors[start_idx:end_idx, 0]**2))
        pos_rms_y = np.sqrt(np.mean(pos_errors[start_idx:end_idx, 1]**2))
        pos_rms.append(np.sqrt(pos_rms_x**2 + pos_rms_y**2))
        
        vel_rms_x = np.sqrt(np.mean(vel_errors[start_idx:end_idx, 0]**2))
        vel_rms_y = np.sqrt(np.mean(vel_errors[start_idx:end_idx, 1]**2))
        vel_rms.append(np.sqrt(vel_rms_x**2 + vel_rms_y**2))
    
    ax.plot(window_times, pos_rms, 'r-', linewidth=2, label='Position RMS Error')
    ax.plot(window_times, vel_rms, 'b-', linewidth=2, label='Velocity RMS Error')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RMS Error')
    ax.set_title('RMS Tracking Errors (10s sliding window)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=10, color='orange', linestyle=':', alpha=0.7, label='Phase transitions')
    ax.axvline(x=20, color='orange', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('imm_rms_errors.png', dpi=300, bbox_inches='tight')
    plt.close()


def print_summary_statistics(pos_errors, vel_errors, acc_errors):
    """Print summary statistics of tracking performance."""
    print("\n" + "="*50)
    print("TRACKING PERFORMANCE SUMMARY")
    print("="*50)
    
    # Overall RMS errors
    pos_rms_x = np.sqrt(np.mean(pos_errors[:, 0]**2))
    pos_rms_y = np.sqrt(np.mean(pos_errors[:, 1]**2))
    pos_rms_total = np.sqrt(pos_rms_x**2 + pos_rms_y**2)
    
    vel_rms_x = np.sqrt(np.mean(vel_errors[:, 0]**2))
    vel_rms_y = np.sqrt(np.mean(vel_errors[:, 1]**2))
    vel_rms_total = np.sqrt(vel_rms_x**2 + vel_rms_y**2)
    
    acc_rms_x = np.sqrt(np.mean(acc_errors[:, 0]**2))
    acc_rms_y = np.sqrt(np.mean(acc_errors[:, 1]**2))
    acc_rms_total = np.sqrt(acc_rms_x**2 + acc_rms_y**2)
    
    print(f"Overall RMS Errors:")
    print(f"  Position: {pos_rms_total:.3f} m (X: {pos_rms_x:.3f} m, Y: {pos_rms_y:.3f} m)")
    print(f"  Velocity: {vel_rms_total:.3f} m/s (X: {vel_rms_x:.3f} m/s, Y: {vel_rms_y:.3f} m/s)")
    print(f"  Acceleration: {acc_rms_total:.3f} m/s² (X: {acc_rms_x:.3f} m/s², Y: {acc_rms_y:.3f} m/s²)")
    print()
    
    # Phase-specific errors
    static_idx = slice(0, 20)  # 0-10s at 0.5s intervals
    accel_idx = slice(20, 40)  # 10-20s
    cv_idx = slice(40, None)   # 20-100s
    
    phases = [("Static Phase (0-10s)", static_idx), 
              ("Acceleration Phase (10-20s)", accel_idx), 
              ("Constant Velocity Phase (20-100s)", cv_idx)]
    
    for phase_name, idx in phases:
        pos_rms = np.sqrt(np.mean(pos_errors[idx, 0]**2 + pos_errors[idx, 1]**2))
        vel_rms = np.sqrt(np.mean(vel_errors[idx, 0]**2 + vel_errors[idx, 1]**2))
        acc_rms = np.sqrt(np.mean(acc_errors[idx, 0]**2 + acc_errors[idx, 1]**2))
        
        print(f"{phase_name}:")
        print(f"  Position RMS: {pos_rms:.3f} m")
        print(f"  Velocity RMS: {vel_rms:.3f} m/s")
        print(f"  Acceleration RMS: {acc_rms:.3f} m/s²")
        print()


if __name__ == '__main__':
    main()