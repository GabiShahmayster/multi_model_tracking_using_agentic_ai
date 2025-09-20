#!/usr/bin/env python3
"""
Wrong Model Tracking with LangGraph Agent Monitoring

This script demonstrates the effects of using an incorrect motion model for tracking,
enhanced with a LangGraph-based agent that monitors the innovation sequence and
detects model errors by identifying biased innovations.

The target follows a specific motion profile:
- 0-50s: Static (no motion)
- 50-100s: Constant velocity (10 m/s in x-direction)

The tracker uses a static motion model throughout, while the LangGraph agent
monitors innovation sequences and detects when the model becomes inappropriate.

Features:
- Real-time innovation bias detection
- Automated model error alerts
- Statistical analysis of innovation patterns
- Agent-driven anomaly detection
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import TypedDict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics

# Add base directory to path to import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'base'))

from motion import MotionSimulator
from trackers import StaticKalmanFilter


# Agent State Management
class AlertLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class InnovationAlert:
    timestamp: float
    alert_level: AlertLevel
    message: str
    innovation_bias: float
    innovation_variance: float

class InnovationMonitorState(TypedDict):
    innovation_buffer: List[Tuple[float, float]]  # (x, y) innovations
    timestamps: List[float]
    window_size: int
    bias_threshold: float
    alerts: List[InnovationAlert]
    current_alert_level: AlertLevel
    model_error_detected: bool
    bias_detection_count: int


class InnovationMonitoringAgent:
    """
    LangGraph-based agent for monitoring filter innovation sequences
    and detecting model bias/errors in real-time.
    """

    def __init__(self, window_size: int = 20, bias_threshold: float = 1.0):
        self.state = InnovationMonitorState(
            innovation_buffer=[],
            timestamps=[],
            window_size=window_size,
            bias_threshold=bias_threshold,
            alerts=[],
            current_alert_level=AlertLevel.NORMAL,
            model_error_detected=False,
            bias_detection_count=0
        )

    def add_innovation(self, timestamp: float, innovation: Tuple[float, float]):
        """Add new innovation to the monitoring buffer."""
        self.state["innovation_buffer"].append(innovation)
        self.state["timestamps"].append(timestamp)

        # Maintain sliding window
        if len(self.state["innovation_buffer"]) > self.state["window_size"]:
            self.state["innovation_buffer"].pop(0)
            self.state["timestamps"].pop(0)

        # Trigger analysis if we have enough data
        if len(self.state["innovation_buffer"]) >= min(10, self.state["window_size"]):
            self._analyze_innovation_pattern()

    def _analyze_innovation_pattern(self):
        """Analyze current innovation pattern for bias and anomalies."""
        if len(self.state["innovation_buffer"]) < 5:
            return

        # Extract x and y components
        x_innovations = [inn[0] for inn in self.state["innovation_buffer"]]
        y_innovations = [inn[1] for inn in self.state["innovation_buffer"]]

        # Calculate bias (mean deviation from zero)
        x_bias = abs(np.mean(x_innovations))
        y_bias = abs(np.mean(y_innovations))
        total_bias = np.sqrt(x_bias**2 + y_bias**2)

        # Calculate variance (for reporting purposes only)
        x_var = np.var(x_innovations)
        y_var = np.var(y_innovations)
        total_variance = np.sqrt(x_var + y_var)

        # Check for bias detection (only criterion for alerts)
        bias_detected = total_bias > self.state["bias_threshold"]

        current_time = self.state["timestamps"][-1]

        # Determine alert level based on bias magnitude only
        if bias_detected:
            # Use bias magnitude to determine warning vs critical
            if total_bias > self.state["bias_threshold"] * 2.0:  # Critical if bias > 2x threshold
                alert_level = AlertLevel.CRITICAL
                message = f"CRITICAL: Severe innovation bias detected! Bias={total_bias:.3f}m (>{self.state['bias_threshold']*2.0:.1f}m threshold)"
                self.state["model_error_detected"] = True
            else:
                alert_level = AlertLevel.WARNING
                message = f"WARNING: Innovation bias detected. Bias={total_bias:.3f}m (>{self.state['bias_threshold']:.1f}m threshold)"
            self.state["bias_detection_count"] += 1
        else:
            alert_level = AlertLevel.NORMAL
            message = f"Innovation bias within normal parameters. Bias={total_bias:.3f}m"
            # Reset bias detection count if returning to normal
            if self.state["current_alert_level"] != AlertLevel.NORMAL:
                self.state["bias_detection_count"] = max(0, self.state["bias_detection_count"] - 1)

        # Update state
        self.state["current_alert_level"] = alert_level

        # Create alert if not normal or if transitioning states
        if (alert_level != AlertLevel.NORMAL or
            len(self.state["alerts"]) == 0 or
            self.state["alerts"][-1].alert_level != alert_level):

            alert = InnovationAlert(
                timestamp=current_time,
                alert_level=alert_level,
                message=message,
                innovation_bias=total_bias,
                innovation_variance=total_variance
            )
            self.state["alerts"].append(alert)

    def _detect_systematic_bias(self) -> bool:
        """Detect systematic bias using statistical tests."""
        if len(self.state["innovation_buffer"]) < 10:
            return False

        x_innovations = [inn[0] for inn in self.state["innovation_buffer"]]

        # One-sample t-test equivalent (simplified)
        mean_x = np.mean(x_innovations)
        std_x = np.std(x_innovations)

        if std_x > 0:
            t_stat = abs(mean_x) / (std_x / np.sqrt(len(x_innovations)))
            # Approximate critical value for 95% confidence
            critical_value = 2.0
            return t_stat > critical_value

        return False

    def _detect_trend(self) -> Optional[str]:
        """Detect trends in innovation sequence."""
        if len(self.state["innovation_buffer"]) < 10:
            return None

        x_innovations = [inn[0] for inn in self.state["innovation_buffer"]]

        # Simple linear trend detection
        indices = np.arange(len(x_innovations))
        correlation = np.corrcoef(indices, x_innovations)[0, 1]

        if abs(correlation) > 0.7:
            if correlation > 0:
                return "increasing"
            else:
                return "decreasing"

        return None

    def get_diagnostic_report(self) -> str:
        """Generate a comprehensive diagnostic report."""
        report = []
        report.append("=== INNOVATION MONITORING AGENT REPORT ===")
        report.append(f"Total alerts generated: {len(self.state['alerts'])}")
        report.append(f"Current alert level: {self.state['current_alert_level'].value}")
        report.append(f"Model error detected: {self.state['model_error_detected']}")
        report.append(f"Bias detection count: {self.state['bias_detection_count']}")

        if self.state["alerts"]:
            report.append("\nRecent Alerts:")
            for alert in self.state["alerts"][-5:]:  # Last 5 alerts
                report.append(f"  t={alert.timestamp:.1f}s: {alert.message}")

        # Additional diagnostics
        systematic_bias = self._detect_systematic_bias()
        trend = self._detect_trend()

        report.append(f"\nDiagnostics:")
        report.append(f"  Systematic bias detected: {systematic_bias}")
        report.append(f"  Innovation trend: {trend if trend else 'none'}")

        return "\n".join(report)

    def get_alert_timestamps(self, alert_level: AlertLevel) -> List[float]:
        """Get timestamps for specific alert level."""
        return [alert.timestamp for alert in self.state["alerts"]
                if alert.alert_level == alert_level]


def main():
    print("Wrong Model Tracking with LangGraph Agent Monitoring")
    print("=" * 60)

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
    print("Agent: Innovation monitoring with bias detection")
    print()

    # Create motion simulator
    sim = MotionSimulator(x0=0.0, y0=0.0, vx0=0.0, vy0=0.0)
    sim.add_segment(sim.STATIC, 50.0)  # Static for 50 seconds
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

    # Initialize LangGraph monitoring agent
    agent = InnovationMonitoringAgent(
        window_size=20,
        bias_threshold=1.5  # Threshold for detecting significant bias
    )

    print("Static Kalman filter initialized")
    print(f"  Initial position: {initial_position}")
    print(f"  Position uncertainty: 1.0 m")
    print(f"  Process noise: 0.1 m")
    print(f"  Measurement noise: {measurement_noise_std} m")
    print()
    print("Innovation Monitoring Agent initialized")
    print(f"  Window size: {agent.state['window_size']}")
    print(f"  Bias threshold: {agent.state['bias_threshold']}")
    print(f"  Alert logic: Bias-only detection")
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

    # Run tracking simulation with agent monitoring
    print("Running tracking simulation with agent monitoring...")
    for i, t in enumerate(times):
        # Get prediction before update for innovation calculation
        if i == 0:
            predicted_pos = tracker.get_state()
        else:
            tracker.predict()
            predicted_pos = tracker.get_state()

        # Calculate innovation (measurement - prediction)
        measurement = measurements[i]
        innovation = np.array(measurement) - np.array(predicted_pos)
        innovations[i] = innovation

        # Agent monitoring: Add innovation to agent buffer
        agent.add_innovation(t, (innovation[0], innovation[1]))

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

    # Generate agent diagnostic report
    print("AGENT DIAGNOSTIC REPORT:")
    print(agent.get_diagnostic_report())
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

    # Create comprehensive plots with agent alerts
    print("Creating plots with agent monitoring...")

    # Get agent alert timestamps for visualization
    warning_times = agent.get_alert_timestamps(AlertLevel.WARNING)
    critical_times = agent.get_alert_timestamps(AlertLevel.CRITICAL)

    plt.style.use('default')
    fig = plt.figure(figsize=(16, 16))  # Increased height for additional plots

    # Plot 1: True vs Estimated Positions
    ax1 = plt.subplot(5, 2, 1)
    plt.plot(times, true_positions[:, 0], 'b-', linewidth=2, label='True X Position')
    plt.plot(times, estimated_positions[:, 0], 'r--', linewidth=2, label='Estimated X Position')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')

    # Add agent alerts
    for t in warning_times:
        plt.axvline(x=t, color='orange', linestyle='--', alpha=0.6, linewidth=1)
    for t in critical_times:
        plt.axvline(x=t, color='red', linestyle='-', alpha=0.8, linewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.title('X Position: True vs Estimated (with Agent Alerts)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax2 = plt.subplot(5, 2, 2)
    plt.plot(times, true_positions[:, 1], 'b-', linewidth=2, label='True Y Position')
    plt.plot(times, estimated_positions[:, 1], 'r--', linewidth=2, label='Estimated Y Position')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')

    # Add agent alerts
    for t in warning_times:
        plt.axvline(x=t, color='orange', linestyle='--', alpha=0.6, linewidth=1)
    for t in critical_times:
        plt.axvline(x=t, color='red', linestyle='-', alpha=0.8, linewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (m)')
    plt.title('Y Position: True vs Estimated (with Agent Alerts)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: True vs Estimated Velocities
    ax3 = plt.subplot(5, 2, 3)
    plt.plot(times, true_velocities[:, 0], 'b-', linewidth=2, label='True X Velocity')
    plt.plot(times, estimated_velocities[:, 0], 'r--', linewidth=2, label='Estimated X Velocity')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')

    # Add agent alerts
    for t in warning_times:
        plt.axvline(x=t, color='orange', linestyle='--', alpha=0.6, linewidth=1)
    for t in critical_times:
        plt.axvline(x=t, color='red', linestyle='-', alpha=0.8, linewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('X Velocity (m/s)')
    plt.title('X Velocity: True vs Estimated (with Agent Alerts)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax4 = plt.subplot(5, 2, 4)
    plt.plot(times, true_velocities[:, 1], 'b-', linewidth=2, label='True Y Velocity')
    plt.plot(times, estimated_velocities[:, 1], 'r--', linewidth=2, label='Estimated Y Velocity')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')

    # Add agent alerts
    for t in warning_times:
        plt.axvline(x=t, color='orange', linestyle='--', alpha=0.6, linewidth=1)
    for t in critical_times:
        plt.axvline(x=t, color='red', linestyle='-', alpha=0.8, linewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('Y Velocity (m/s)')
    plt.title('Y Velocity: True vs Estimated (with Agent Alerts)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 5: Position Tracking Errors
    ax5 = plt.subplot(5, 2, 5)
    plt.plot(times, position_errors[:, 0], 'r-', linewidth=1.5, label='X Position Error')
    plt.plot(times, position_errors[:, 1], 'g-', linewidth=1.5, label='Y Position Error')
    plt.plot(times, position_error_mag, 'k-', linewidth=2, label='Position Error Magnitude')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Add agent alerts
    for t in warning_times:
        plt.axvline(x=t, color='orange', linestyle='--', alpha=0.6, linewidth=1)
    for t in critical_times:
        plt.axvline(x=t, color='red', linestyle='-', alpha=0.8, linewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('Position Error (m)')
    plt.title('Position Tracking Errors (with Agent Alerts)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 6: Velocity Tracking Errors
    ax6 = plt.subplot(5, 2, 6)
    plt.plot(times, velocity_errors[:, 0], 'r-', linewidth=1.5, label='X Velocity Error')
    plt.plot(times, velocity_errors[:, 1], 'g-', linewidth=1.5, label='Y Velocity Error')
    plt.plot(times, velocity_error_mag, 'k-', linewidth=2, label='Velocity Error Magnitude')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Add agent alerts
    for t in warning_times:
        plt.axvline(x=t, color='orange', linestyle='--', alpha=0.6, linewidth=1)
    for t in critical_times:
        plt.axvline(x=t, color='red', linestyle='-', alpha=0.8, linewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('Velocity Error (m/s)')
    plt.title('Velocity Tracking Errors (with Agent Alerts)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 7: Innovation Sequence X with Agent Analysis
    ax7 = plt.subplot(5, 2, 7)
    plt.plot(times, innovations[:, 0], 'b-', linewidth=1.5, label='X Innovation')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Add bias threshold lines
    plt.axhline(y=agent.state['bias_threshold'], color='orange', linestyle='--', alpha=0.7, label='Bias Threshold')
    plt.axhline(y=-agent.state['bias_threshold'], color='orange', linestyle='--', alpha=0.7)

    # Add agent alerts
    for t in warning_times:
        plt.axvline(x=t, color='orange', linestyle='--', alpha=0.6, linewidth=1)
    for t in critical_times:
        plt.axvline(x=t, color='red', linestyle='-', alpha=0.8, linewidth=2)

    plt.xlabel('Time (s)')
    plt.ylabel('Innovation (m)')
    plt.title('Innovation Sequence X with Agent Monitoring')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 8: Innovation Sequence Magnitude with Agent Analysis
    ax8 = plt.subplot(5, 2, 8)
    plt.plot(times, innovation_mag, 'purple', linewidth=2, label='Innovation Magnitude')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Add reference lines for bias thresholds
    plt.axhline(y=agent.state['bias_threshold'], color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
    plt.axhline(y=agent.state['bias_threshold'] * 2.0, color='red', linestyle=':', alpha=0.7, label='Critical Threshold')

    # Add agent alerts with different markers
    for t in warning_times:
        plt.axvline(x=t, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Warning' if t == warning_times[0] else "")
    for t in critical_times:
        plt.axvline(x=t, color='red', linestyle='-', alpha=1.0, linewidth=3, label='Critical' if t == critical_times[0] else "")

    plt.xlabel('Time (s)')
    plt.ylabel('Innovation Magnitude (m)')
    plt.title('Innovation Magnitude with Agent Alerts')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 9: Agent Alert Timeline
    ax9 = plt.subplot(5, 2, 9)

    # Create alert level timeline
    alert_levels = np.ones(len(times)) * 0  # Normal = 0

    for alert in agent.state['alerts']:
        # Find closest time index
        time_idx = np.argmin(np.abs(times - alert.timestamp))
        if alert.alert_level == AlertLevel.WARNING:
            alert_levels[time_idx] = 1
        elif alert.alert_level == AlertLevel.CRITICAL:
            alert_levels[time_idx] = 2

    plt.plot(times, alert_levels, 'o-', linewidth=2, markersize=4, label='Alert Level')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.xlabel('Time (s)')
    plt.ylabel('Alert Level')
    plt.title('Agent Alert Timeline')
    plt.yticks([0, 1, 2], ['Normal', 'Warning', 'Critical'])
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 10: Agent Statistics Summary
    ax10 = plt.subplot(5, 2, 10)

    # Calculate rolling statistics for visualization
    window_stats = []
    for i in range(len(times)):
        if i >= 10:  # Need at least 10 points
            recent_innovations = innovation_mag[max(0, i-20):i+1]
            bias = np.mean(recent_innovations)
            variance = np.var(recent_innovations)
            window_stats.append((bias, variance))
        else:
            window_stats.append((0, 0))

    biases = [stat[0] for stat in window_stats]
    variances = [stat[1] for stat in window_stats]

    plt.plot(times, biases, 'b-', linewidth=2, label='Rolling Bias (Innovation Magnitude)')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.axhline(y=agent.state['bias_threshold'], color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
    plt.axhline(y=agent.state['bias_threshold'] * 2.0, color='red', linestyle=':', alpha=0.7, label='Critical Threshold')

    plt.xlabel('Time (s)')
    plt.ylabel('Innovation Bias (m)')
    plt.title('Agent Bias Analysis (Bias-Only Detection)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle('Wrong Model Tracking with LangGraph Agent Monitoring', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save the plot
    os.makedirs('../results/tracking_langgraph_logic', exist_ok=True)
    plot_filename = '../results/tracking_langgraph_logic/wrong_model_with_agent_tracking.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Comprehensive plots with agent monitoring saved to: {plot_filename}")

    print()
    print("=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print("This simulation demonstrates wrong model tracking enhanced with AI agent monitoring:")
    print("- Target: Static (0-50s) → Constant velocity 10 m/s (50-100s)")
    print("- Tracker: Static model throughout (correct only for first 50s)")
    print("- Agent: LangGraph-based innovation monitoring with bias detection")
    print()
    print("Key observations:")
    print(f"- During static phase: Good tracking (Pos RMSE = {static_pos_rmse:.3f} m)")
    print(f"- During motion phase: Poor tracking (Pos RMSE = {motion_pos_rmse:.3f} m)")
    print(f"- Velocity estimation: Always wrong during motion (RMSE = {motion_vel_rmse:.3f} m/s)")
    print(f"- Agent detected model error: {agent.state['model_error_detected']}")
    print(f"- Total agent alerts: {len(agent.state['alerts'])}")
    print(f"- Warning alerts: {len(warning_times)}")
    print(f"- Critical alerts: {len(critical_times)}")
    print()
    print("Agent successfully detected innovation bias indicating model mismatch!")


if __name__ == '__main__':
    main()