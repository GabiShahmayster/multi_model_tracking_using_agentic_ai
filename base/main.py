#!/usr/bin/env python3
"""
Main simulation script for specific motion sequence:
- 0-10s: Static (no motion)
- 10-20s: Constant acceleration (1 m/s² in x-direction)
- 20-100s: Constant velocity (maintaining velocity from acceleration phase)

Generates position and velocity plots saved to files.
"""

import numpy as np
import matplotlib.pyplot as plt
from motion import MotionSimulator


def main():
    print("Motion Simulation: 100-second sequence")
    print("=" * 50)
    
    # Create motion simulator starting at origin with zero velocity
    sim = MotionSimulator(x0=0.0, y0=0.0, vx0=0.0, vy0=0.0)
    
    # Define the motion sequence
    print("Motion sequence:")
    print("  0-10s:  Static (no motion)")
    print(" 10-20s:  Constant acceleration (1 m/s² in x-direction)")
    print(" 20-100s: Constant velocity")
    print()
    
    # Add motion segments
    sim.add_segment(sim.STATIC, 10.0)  # Static for 10 seconds
    sim.add_segment(sim.CONSTANT_ACCELERATION, 10.0, ax=1.0, ay=0.0)  # 1 m/s² x-acceleration for 10 seconds
    sim.add_segment(sim.CONSTANT_VELOCITY, 80.0)  # Constant velocity for 80 seconds
    
    print(f"Total simulation duration: {sim.duration} seconds")
    print(f"Number of segments: {len(sim.segments)}")
    print()
    
    # Calculate key values at segment boundaries
    print("Key values at segment boundaries:")
    # At t=10s (end of static phase)
    pos_10 = sim.get_position(10.0)
    vel_10 = sim.get_velocity(10.0)
    print(f"t=10s: pos=({pos_10[0]:.2f}, {pos_10[1]:.2f}), vel=({vel_10[0]:.2f}, {vel_10[1]:.2f})")
    
    # At t=20s (end of acceleration phase)
    pos_20 = sim.get_position(20.0)
    vel_20 = sim.get_velocity(20.0)
    print(f"t=20s: pos=({pos_20[0]:.2f}, {pos_20[1]:.2f}), vel=({vel_20[0]:.2f}, {vel_20[1]:.2f})")
    
    # At t=100s (end of simulation)
    pos_100 = sim.get_position(100.0)
    vel_100 = sim.get_velocity(100.0)
    print(f"t=100s: pos=({pos_100[0]:.2f}, {pos_100[1]:.2f}), vel=({vel_100[0]:.2f}, {vel_100[1]:.2f})")
    print()
    
    # Generate detailed motion history for plotting
    print("Generating motion history for plotting...")
    time_step = 0.5  # 0.5 second intervals
    history = sim.get_motion_history(time_step=time_step)
    
    # Extract data for plotting
    times = [entry['time'] for entry in history]
    x_positions = [entry['position'][0] for entry in history]
    y_positions = [entry['position'][1] for entry in history]
    x_velocities = [entry['velocity'][0] for entry in history]
    y_velocities = [entry['velocity'][1] for entry in history]
    
    # Create plots
    print("Creating plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Motion Simulation: 100-second Sequence', fontsize=16, fontweight='bold')
    
    # Plot 1: X-position vs time
    ax1.plot(times, x_positions, 'b-', linewidth=2, label='X Position')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X Position (m)')
    ax1.set_title('X Position vs Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add vertical lines at segment boundaries
    ax1.axvline(x=10, color='r', linestyle='--', alpha=0.7, label='Phase transitions')
    ax1.axvline(x=20, color='r', linestyle='--', alpha=0.7)
    
    # Plot 2: Y-position vs time
    ax2.plot(times, y_positions, 'g-', linewidth=2, label='Y Position')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Y Position vs Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add vertical lines at segment boundaries
    ax2.axvline(x=10, color='r', linestyle='--', alpha=0.7, label='Phase transitions')
    ax2.axvline(x=20, color='r', linestyle='--', alpha=0.7)
    
    # Plot 3: X-velocity vs time
    ax3.plot(times, x_velocities, 'b-', linewidth=2, label='X Velocity')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('X Velocity (m/s)')
    ax3.set_title('X Velocity vs Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add vertical lines at segment boundaries
    ax3.axvline(x=10, color='r', linestyle='--', alpha=0.7, label='Phase transitions')
    ax3.axvline(x=20, color='r', linestyle='--', alpha=0.7)
    
    # Plot 4: Y-velocity vs time
    ax4.plot(times, y_velocities, 'g-', linewidth=2, label='Y Velocity')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Y Velocity (m/s)')
    ax4.set_title('Y Velocity vs Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add vertical lines at segment boundaries
    ax4.axvline(x=10, color='r', linestyle='--', alpha=0.7, label='Phase transitions')
    ax4.axvline(x=20, color='r', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plot_filename = 'motion_simulation_plots.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_filename}")
    
    # Create a separate trajectory plot
    plt.figure(figsize=(10, 8))
    plt.plot(x_positions, y_positions, 'b-', linewidth=2, label='Trajectory')
    plt.scatter(x_positions[0], y_positions[0], color='green', s=100, label='Start', zorder=5)
    plt.scatter(x_positions[-1], y_positions[-1], color='red', s=100, label='End', zorder=5)
    
    # Mark key points
    idx_10s = int(10 / time_step)
    idx_20s = int(20 / time_step)
    plt.scatter(x_positions[idx_10s], y_positions[idx_10s], color='orange', s=80, label='t=10s', zorder=5)
    plt.scatter(x_positions[idx_20s], y_positions[idx_20s], color='purple', s=80, label='t=20s', zorder=5)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('2D Trajectory')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    
    # Save trajectory plot
    trajectory_filename = 'motion_trajectory.png'
    plt.savefig(trajectory_filename, dpi=300, bbox_inches='tight')
    print(f"Trajectory plot saved to: {trajectory_filename}")
    
    # Display final summary
    print()
    print("Simulation Summary:")
    print(f"  Final position: ({pos_100[0]:.2f}, {pos_100[1]:.2f}) m")
    print(f"  Final velocity: ({vel_100[0]:.2f}, {vel_100[1]:.2f}) m/s")
    print(f"  Maximum X velocity: {max(x_velocities):.2f} m/s")
    print(f"  Distance traveled: {pos_100[0]:.2f} m")
    
    # Calculate theoretical values for verification
    print()
    print("Theoretical verification:")
    # After 10s acceleration at 1 m/s²: v = at = 1*10 = 10 m/s
    # Distance during acceleration: s = 0.5*a*t² = 0.5*1*10² = 50 m
    # Distance during constant velocity: s = v*t = 10*80 = 800 m
    # Total distance: 50 + 800 = 850 m
    print(f"  Expected final velocity: 10.00 m/s (actual: {vel_100[0]:.2f} m/s)")
    print(f"  Expected final position: 850.00 m (actual: {pos_100[0]:.2f} m)")
    
    print()
    print("Simulation completed successfully!")


if __name__ == '__main__':
    main()