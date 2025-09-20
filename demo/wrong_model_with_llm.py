#!/usr/bin/env python3
"""
Wrong Model Tracking with LLM-Enhanced LangGraph Agent Monitoring

This script demonstrates the effects of using an incorrect motion model for tracking,
enhanced with a LangGraph-based agent that uses local LLM (via Ollama) to analyze
innovation sequences and provide semantic understanding of model errors.

The target follows a specific motion profile:
- 0-50s: Static (no motion)
- 50-100s: Constant velocity (10 m/s in x-direction)

The tracker uses a static motion model throughout, while the LLM-enhanced LangGraph agent
monitors innovation sequences and provides natural language analysis of filter behavior.

Features:
- Multi-node LangGraph workflow with LLM reasoning
- Ollama local LLM integration for pattern analysis
- Statistical + semantic analysis fusion
- Natural language explanations for model errors
- Enhanced alert generation with confidence scoring
"""

import numpy as np
import matplotlib.pyplot as plt
from motion import MotionSimulator
from trackers import StaticKalmanFilter
from typing import TypedDict, List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import statistics
import json

try:
    from langchain_community.llms import Ollama
    from langgraph.graph import StateGraph, END
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("Warning: LangChain/LangGraph not available. Using mock LLM implementation.")
    LANGCHAIN_AVAILABLE = False

    # Mock classes for when LangGraph is not available
    class StateGraph:
        def __init__(self, state_type):
            self.state_type = state_type
            self.nodes = {}
            self.edges = []
            self.entry_point = None

        def add_node(self, name, func):
            self.nodes[name] = func

        def add_edge(self, from_node, to_node):
            self.edges.append((from_node, to_node))

        def set_entry_point(self, node):
            self.entry_point = node

        def compile(self):
            return MockWorkflow(self.nodes, self.edges, self.entry_point)

    class MockWorkflow:
        def __init__(self, nodes, edges, entry_point):
            self.nodes = nodes
            self.edges = edges
            self.entry_point = entry_point

        def invoke(self, state):
            # Execute all nodes in sequence for mock implementation
            current_state = state.copy()

            # Execute workflow: statistical_analysis -> llm_reasoning -> decision_fusion -> alert_generation
            node_sequence = ["statistical_analysis", "llm_reasoning", "decision_fusion", "alert_generation"]

            for node_name in node_sequence:
                if node_name in self.nodes:
                    current_state = self.nodes[node_name](current_state)

            return current_state

    END = "END"


# Enhanced State Management with LLM capabilities
class AlertLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class EnhancedInnovationAlert:
    timestamp: float
    alert_level: AlertLevel
    message: str
    innovation_bias: float
    innovation_variance: float
    llm_analysis: str
    confidence_score: float
    pattern_description: str

class LLMEnhancedMonitorState(TypedDict):
    innovation_buffer: List[Tuple[float, float]]  # (x, y) innovations
    timestamps: List[float]
    window_size: int
    bias_threshold: float
    alerts: List[EnhancedInnovationAlert]
    current_alert_level: AlertLevel
    model_error_detected: bool
    bias_detection_count: int

    # LLM-specific state
    llm_conversation_history: List[Dict[str, str]]
    pattern_descriptions: List[str]
    confidence_scores: List[float]
    llm_recommendations: List[str]
    statistical_summary: Dict[str, float]


class MockOllama:
    """Mock LLM for when Ollama is not available"""

    def __call__(self, prompt: str) -> str:
        # Simple pattern-based responses for demonstration
        if "sudden increase" in prompt.lower() or "bias" in prompt.lower():
            return ("Analysis: The innovation sequence shows a systematic bias indicating "
                   "model mismatch. The static filter cannot capture the target's motion, "
                   "resulting in persistent prediction errors. Confidence: High (0.85). "
                   "Recommendation: Switch to constant velocity model.")
        elif "normal" in prompt.lower() or "small" in prompt.lower():
            return ("Analysis: Innovation sequence appears normal with expected measurement "
                   "noise characteristics. No systematic bias detected. "
                   "Confidence: Medium (0.70). Recommendation: Continue current model.")
        else:
            return ("Analysis: Analyzing innovation patterns for model appropriateness. "
                   "Statistical indicators suggest careful monitoring is needed. "
                   "Confidence: Medium (0.60).")


class LLMEnhancedInnovationAgent:
    """
    LangGraph-based agent with LLM integration for semantic innovation analysis
    """

    def __init__(self, window_size: int = 20, bias_threshold: float = 1.5,
                 ollama_model: str = "llama3.2"):

        # Initialize LLM
        if LANGCHAIN_AVAILABLE:
            try:
                self.llm = Ollama(model=ollama_model, base_url="http://localhost:11434")
                print(f"Initialized Ollama with model: {ollama_model}")
            except Exception as e:
                print(f"Failed to connect to Ollama: {e}")
                print("Using mock LLM implementation")
                self.llm = MockOllama()
        else:
            self.llm = MockOllama()

        # Initialize state
        self.state = LLMEnhancedMonitorState(
            innovation_buffer=[],
            timestamps=[],
            window_size=window_size,
            bias_threshold=bias_threshold,
            alerts=[],
            current_alert_level=AlertLevel.NORMAL,
            model_error_detected=False,
            bias_detection_count=0,
            llm_conversation_history=[],
            pattern_descriptions=[],
            confidence_scores=[],
            llm_recommendations=[],
            statistical_summary={}
        )

        # Build LangGraph workflow
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the multi-node LangGraph workflow"""

        # Define the workflow graph
        workflow = StateGraph(LLMEnhancedMonitorState)

        # Add nodes
        workflow.add_node("statistical_analysis", self._statistical_analysis_node)
        workflow.add_node("llm_reasoning", self._llm_reasoning_node)
        workflow.add_node("decision_fusion", self._decision_fusion_node)
        workflow.add_node("alert_generation", self._alert_generation_node)

        # Define the flow
        workflow.set_entry_point("statistical_analysis")
        workflow.add_edge("statistical_analysis", "llm_reasoning")
        workflow.add_edge("llm_reasoning", "decision_fusion")
        workflow.add_edge("decision_fusion", "alert_generation")
        workflow.add_edge("alert_generation", END)

        return workflow.compile()

    def _statistical_analysis_node(self, state: LLMEnhancedMonitorState) -> LLMEnhancedMonitorState:
        """Node 1: Perform statistical analysis of innovation sequence"""

        if len(state["innovation_buffer"]) < 5:
            return state

        # Extract components
        x_innovations = [inn[0] for inn in state["innovation_buffer"]]
        y_innovations = [inn[1] for inn in state["innovation_buffer"]]

        # Calculate statistics
        x_bias = abs(np.mean(x_innovations))
        y_bias = abs(np.mean(y_innovations))
        total_bias = np.sqrt(x_bias**2 + y_bias**2)

        x_var = np.var(x_innovations)
        y_var = np.var(y_innovations)
        total_var = np.sqrt(x_var + y_var)

        # Trend analysis
        if len(x_innovations) >= 10:
            x_trend = np.polyfit(range(len(x_innovations)), x_innovations, 1)[0]
            y_trend = np.polyfit(range(len(y_innovations)), y_innovations, 1)[0]
        else:
            x_trend = 0.0
            y_trend = 0.0

        # Update statistical summary
        state["statistical_summary"] = {
            "x_bias": x_bias,
            "y_bias": y_bias,
            "total_bias": total_bias,
            "x_variance": x_var,
            "y_variance": y_var,
            "total_variance": total_var,
            "x_trend": x_trend,
            "y_trend": y_trend,
            "sample_count": len(x_innovations)
        }

        return state

    def _llm_reasoning_node(self, state: LLMEnhancedMonitorState) -> LLMEnhancedMonitorState:
        """Node 2: LLM analysis of innovation patterns"""

        stats = state["statistical_summary"]
        if not stats:
            return state

        # Prepare innovation data for LLM
        recent_innovations = state["innovation_buffer"][-10:] if len(state["innovation_buffer"]) >= 10 else state["innovation_buffer"]
        innovation_summary = {
            "recent_innovations": recent_innovations,
            "statistical_metrics": stats
        }

        # Create structured prompt for LLM
        prompt = self._create_llm_prompt(innovation_summary, state["timestamps"][-1] if state["timestamps"] else 0.0)

        # Get LLM analysis
        try:
            llm_response = self.llm(prompt)

            # Parse LLM response (simplified parsing)
            analysis_parts = llm_response.split("Confidence:")
            analysis_text = analysis_parts[0].strip()

            if len(analysis_parts) > 1:
                confidence_part = analysis_parts[1].split("Recommendation:")[0].strip()
                try:
                    # Extract confidence score
                    confidence_score = float(''.join(filter(str.isdigit, confidence_part.split('(')[1].split(')')[0]))) / 100.0
                except:
                    confidence_score = 0.5
            else:
                confidence_score = 0.5

            # Extract recommendation
            if "Recommendation:" in llm_response:
                recommendation = llm_response.split("Recommendation:")[1].strip()
            else:
                recommendation = "Continue monitoring"

        except Exception as e:
            print(f"LLM analysis failed: {e}")
            analysis_text = "LLM analysis unavailable"
            confidence_score = 0.0
            recommendation = "Statistical analysis only"

        # Update state with LLM insights
        state["llm_conversation_history"].append({
            "timestamp": state["timestamps"][-1] if state["timestamps"] else 0.0,
            "prompt": prompt,
            "response": llm_response if 'llm_response' in locals() else analysis_text
        })

        state["confidence_scores"].append(confidence_score)
        state["llm_recommendations"].append(recommendation)

        # Store analysis for decision fusion
        state["current_llm_analysis"] = analysis_text
        state["current_confidence"] = confidence_score

        return state

    def _decision_fusion_node(self, state: LLMEnhancedMonitorState) -> LLMEnhancedMonitorState:
        """Node 3: Fuse statistical and LLM analysis for decision making"""

        stats = state["statistical_summary"]
        if not stats:
            return state

        # Statistical decision
        bias_detected = stats["total_bias"] > state["bias_threshold"]
        high_variance = stats["total_variance"] > 2.0

        # LLM confidence weighting
        llm_confidence = state.get("current_confidence", 0.5)
        llm_suggests_error = "bias" in state.get("current_llm_analysis", "").lower() or "mismatch" in state.get("current_llm_analysis", "").lower()

        # Fusion logic
        confidence_weighted_bias = bias_detected and (llm_confidence > 0.6)
        llm_reinforced_detection = bias_detected and llm_suggests_error

        # Final decision
        model_error_detected = bias_detected or (llm_suggests_error and llm_confidence > 0.7)

        # Determine alert level
        if model_error_detected:
            if stats["total_bias"] > state["bias_threshold"] * 2.0 or llm_confidence > 0.8:
                alert_level = AlertLevel.CRITICAL
            else:
                alert_level = AlertLevel.WARNING
        else:
            alert_level = AlertLevel.NORMAL

        # Update state
        state["model_error_detected"] = model_error_detected
        state["current_alert_level"] = alert_level
        if bias_detected:
            state["bias_detection_count"] += 1

        return state

    def _alert_generation_node(self, state: LLMEnhancedMonitorState) -> LLMEnhancedMonitorState:
        """Node 4: Generate enhanced alerts with LLM insights"""

        if state["current_alert_level"] == AlertLevel.NORMAL:
            return state

        stats = state["statistical_summary"]
        timestamp = state["timestamps"][-1] if state["timestamps"] else 0.0

        # Create enhanced alert
        if state["current_alert_level"] == AlertLevel.CRITICAL:
            message = f"CRITICAL: Severe model mismatch detected! Bias={stats['total_bias']:.3f}m"
        else:
            message = f"WARNING: Potential model error detected. Bias={stats['total_bias']:.3f}m"

        # Add LLM analysis to message
        llm_analysis = state.get("current_llm_analysis", "No LLM analysis available")
        confidence = state.get("current_confidence", 0.0)

        alert = EnhancedInnovationAlert(
            timestamp=timestamp,
            alert_level=state["current_alert_level"],
            message=message,
            innovation_bias=stats["total_bias"],
            innovation_variance=stats["total_variance"],
            llm_analysis=llm_analysis,
            confidence_score=confidence,
            pattern_description=f"Bias: {stats['total_bias']:.3f}m, Variance: {stats['total_variance']:.3f}"
        )

        state["alerts"].append(alert)

        return state

    def _create_llm_prompt(self, innovation_data: Dict, current_time: float) -> str:
        """Create structured prompt for LLM analysis"""

        stats = innovation_data["statistical_metrics"]

        prompt = f"""
You are analyzing innovation sequences from a Kalman filter tracking system at time {current_time:.1f}s.

CONTEXT:
- Filter Model: Static motion model (assumes target is stationary)
- Innovation = Measurement - Prediction (should be zero-mean noise if model is correct)
- Target Motion: Static (0-50s), then Constant Velocity 10 m/s (50-100s)

CURRENT STATISTICS:
- X-axis bias: {stats['x_bias']:.3f}m
- Y-axis bias: {stats['y_bias']:.3f}m
- Total bias magnitude: {stats['total_bias']:.3f}m
- X-axis variance: {stats['x_variance']:.3f}
- Y-axis variance: {stats['y_variance']:.3f}
- Sample count: {stats['sample_count']}

RECENT INNOVATIONS (x, y):
{innovation_data['recent_innovations'][-5:] if len(innovation_data['recent_innovations']) >= 5 else innovation_data['recent_innovations']}

ANALYSIS REQUEST:
1. Evaluate if the innovation pattern indicates model mismatch
2. Consider the expected motion profile vs filter assumptions
3. Assess confidence in your analysis (scale 0.0-1.0)
4. Provide recommendation for filter adaptation

Format your response as:
Analysis: [Your interpretation of the innovation patterns and model appropriateness]
Confidence: [Score 0.0-1.0] ([percentage]%)
Recommendation: [Suggested action for the tracking system]
"""

        return prompt

    def add_innovation(self, timestamp: float, innovation: Tuple[float, float]):
        """Add new innovation and trigger LLM-enhanced analysis"""

        # Update innovation buffer
        self.state["innovation_buffer"].append(innovation)
        self.state["timestamps"].append(timestamp)

        # Maintain sliding window
        if len(self.state["innovation_buffer"]) > self.state["window_size"]:
            self.state["innovation_buffer"].pop(0)
            self.state["timestamps"].pop(0)

        # Trigger LLM workflow if we have enough data
        if len(self.state["innovation_buffer"]) >= min(10, self.state["window_size"]):
            try:
                # Run the complete LangGraph workflow
                self.state = self.workflow.invoke(self.state)
            except Exception as e:
                print(f"Workflow execution failed: {e}")

    def get_diagnostic_report(self) -> str:
        """Generate comprehensive diagnostic report"""

        report = "=== LLM-ENHANCED INNOVATION MONITORING REPORT ===\n"
        report += f"Total alerts generated: {len(self.state['alerts'])}\n"
        report += f"Current alert level: {self.state['current_alert_level'].value}\n"
        report += f"Model error detected: {self.state['model_error_detected']}\n"
        report += f"Bias detection count: {self.state['bias_detection_count']}\n"
        report += f"LLM analyses performed: {len(self.state['llm_conversation_history'])}\n\n"

        if self.state["alerts"]:
            report += "Recent Enhanced Alerts:\n"
            for alert in self.state["alerts"][-5:]:
                report += f"  t={alert.timestamp:.1f}s: {alert.message}\n"
                report += f"    LLM Analysis: {alert.llm_analysis[:100]}...\n"
                report += f"    Confidence: {alert.confidence_score:.2f}\n\n"

        if self.state["llm_recommendations"]:
            report += f"Latest LLM Recommendation: {self.state['llm_recommendations'][-1]}\n"

        return report


def main():
    print("Wrong Model Tracking with LLM-Enhanced LangGraph Agent")
    print("=" * 60)

    # Simulation parameters
    duration = 100.0
    dt = 0.5
    measurement_noise_std = 0.5

    print(f"Simulation duration: {duration} seconds")
    print(f"Time step: {dt} seconds")
    print(f"Measurement noise: {measurement_noise_std} m std dev")
    print()

    print("Target motion profile:")
    print("  0-50s:  Static (no motion)")
    print(" 50-100s: Constant velocity (10 m/s in x-direction)")
    print()
    print("Tracker model: Static motion model (WRONG for 50-100s)")
    print("Agent: LLM-enhanced innovation monitoring with semantic analysis")
    print()

    # Create motion simulator
    sim = MotionSimulator(x0=0.0, y0=0.0, vx0=0.0, vy0=0.0)
    sim.add_segment(sim.STATIC, 50.0)
    sim.add_segment(sim.CONSTANT_ACCELERATION, 0.1, ax=100.0, ay=0.0)
    sim.add_segment(sim.CONSTANT_VELOCITY, 49.9)

    print(f"Motion simulator created with {len(sim.segments)} segments")
    print(f"Total simulation duration: {sim.duration} seconds")
    print()

    # Initialize static Kalman filter
    initial_position = (0.0, 0.0)
    tracker = StaticKalmanFilter(
        initial_position=initial_position,
        position_uncertainty=1.0,
        process_noise=0.1,
        measurement_noise=measurement_noise_std
    )

    print("Static Kalman filter initialized")
    print()

    # Initialize LLM-enhanced agent
    agent = LLMEnhancedInnovationAgent(
        window_size=20,
        bias_threshold=1.5
    )

    print("LLM-enhanced Innovation Monitoring Agent initialized")
    print("  Window size: 20")
    print("  Bias threshold: 1.5")
    print("  LLM integration: Enabled")
    print()

    # Generate simulation data
    times = np.arange(0, duration + dt, dt)
    n_steps = len(times)

    print(f"Generating {n_steps} time steps for simulation...")

    # Arrays for data storage
    true_positions = np.zeros((n_steps, 2))
    true_velocities = np.zeros((n_steps, 2))
    measurements = np.zeros((n_steps, 2))
    estimated_positions = np.zeros((n_steps, 2))
    estimated_velocities = np.zeros((n_steps, 2))
    position_errors = np.zeros((n_steps, 2))
    velocity_errors = np.zeros((n_steps, 2))
    innovations = np.zeros((n_steps, 2))

    # Generate true motion and measurements
    np.random.seed(42)
    for i, t in enumerate(times):
        true_pos = sim.get_position(t)
        true_vel = sim.get_velocity(t)

        true_positions[i] = true_pos
        true_velocities[i] = true_vel

        noise = np.random.normal(0, measurement_noise_std, 2)
        measurements[i] = np.array(true_pos) + noise

    print("True motion and measurements generated")
    print()

    # Run tracking simulation with LLM-enhanced monitoring
    print("Running tracking simulation with LLM-enhanced agent monitoring...")

    for i, t in enumerate(times):
        # Get prediction for innovation calculation
        if i == 0:
            predicted_pos = tracker.get_state()
        else:
            tracker.predict()
            predicted_pos = tracker.get_state()

        # Calculate innovation
        measurement = measurements[i]
        innovation = np.array(measurement) - np.array(predicted_pos)
        innovations[i] = innovation

        # Add innovation to LLM-enhanced agent
        agent.add_innovation(t, tuple(innovation))

        # Update tracker
        tracker.update(measurement)

        # Store estimates and errors
        estimated_pos = tracker.get_state()
        estimated_vel = (0.0, 0.0)

        estimated_positions[i] = estimated_pos
        estimated_velocities[i] = estimated_vel

        position_errors[i] = np.array(true_positions[i]) - np.array(estimated_pos)
        velocity_errors[i] = np.array(true_velocities[i]) - np.array(estimated_vel)

    print("Tracking simulation completed")
    print()

    # Generate LLM-enhanced diagnostic report
    print("LLM-ENHANCED AGENT DIAGNOSTIC REPORT:")
    print(agent.get_diagnostic_report())

    # Calculate error statistics
    position_error_mag = np.sqrt(position_errors[:, 0]**2 + position_errors[:, 1]**2)
    velocity_error_mag = np.sqrt(velocity_errors[:, 0]**2 + velocity_errors[:, 1]**2)

    static_indices = times <= 50.0
    motion_indices = times > 50.0

    static_pos_rmse = np.sqrt(np.mean(position_error_mag[static_indices]**2))
    static_vel_rmse = np.sqrt(np.mean(velocity_error_mag[static_indices]**2))
    motion_pos_rmse = np.sqrt(np.mean(position_error_mag[motion_indices]**2))
    motion_vel_rmse = np.sqrt(np.mean(velocity_error_mag[motion_indices]**2))
    overall_pos_rmse = np.sqrt(np.mean(position_error_mag**2))
    overall_vel_rmse = np.sqrt(np.mean(velocity_error_mag**2))

    print("Error Statistics:")
    print(f"  Static phase (0-50s) - Position RMSE: {static_pos_rmse:.3f} m")
    print(f"  Static phase (0-50s) - Velocity RMSE: {static_vel_rmse:.3f} m/s")
    print(f"  Motion phase (50-100s) - Position RMSE: {motion_pos_rmse:.3f} m")
    print(f"  Motion phase (50-100s) - Velocity RMSE: {motion_vel_rmse:.3f} m/s")
    print(f"  Overall - Position RMSE: {overall_pos_rmse:.3f} m")
    print(f"  Overall - Velocity RMSE: {overall_vel_rmse:.3f} m")
    print()

    # Create enhanced plots with LLM insights
    print("Creating plots with LLM-enhanced agent analysis...")

    plt.style.use('default')
    fig = plt.figure(figsize=(18, 16))

    # Plot 1: True vs Estimated X Positions
    ax1 = plt.subplot(5, 2, 1)
    plt.plot(times, true_positions[:, 0], 'b-', linewidth=2, label='True X Position')
    plt.plot(times, estimated_positions[:, 0], 'r--', linewidth=2, label='Estimated X Position')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.title('X Position: True vs Estimated')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: True vs Estimated Y Positions
    ax2 = plt.subplot(5, 2, 2)
    plt.plot(times, true_positions[:, 1], 'b-', linewidth=2, label='True Y Position')
    plt.plot(times, estimated_positions[:, 1], 'r--', linewidth=2, label='Estimated Y Position')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (m)')
    plt.title('Y Position: True vs Estimated')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Velocity Tracking Errors
    ax3 = plt.subplot(5, 2, 3)
    plt.plot(times, velocity_errors[:, 0], 'r-', linewidth=1.5, label='X Velocity Error')
    plt.plot(times, velocity_errors[:, 1], 'g-', linewidth=1.5, label='Y Velocity Error')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity Error (m/s)')
    plt.title('Velocity Tracking Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Position Tracking Errors
    ax4 = plt.subplot(5, 2, 4)
    plt.plot(times, position_errors[:, 0], 'r-', linewidth=1.5, label='X Position Error')
    plt.plot(times, position_errors[:, 1], 'g-', linewidth=1.5, label='Y Position Error')
    plt.plot(times, position_error_mag, 'k-', linewidth=2, label='Position Error Magnitude')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error (m)')
    plt.title('Position Tracking Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 5: Innovation Sequence X
    ax5 = plt.subplot(5, 2, 5)
    plt.plot(times, innovations[:, 0], 'b-', linewidth=1.5, label='X Innovation')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Innovation (m)')
    plt.title('Innovation Sequence X')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 6: Innovation Sequence Y
    ax6 = plt.subplot(5, 2, 6)
    plt.plot(times, innovations[:, 1], 'g-', linewidth=1.5, label='Y Innovation')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Innovation (m)')
    plt.title('Innovation Sequence Y')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 7: LLM Confidence Scores
    ax7 = plt.subplot(5, 2, 7)
    if agent.state["confidence_scores"] and len(agent.state["confidence_scores"]) > 0:
        # Use the same number of timestamps as confidence scores
        num_scores = len(agent.state["confidence_scores"])
        confidence_times = agent.state["timestamps"][-num_scores:] if len(agent.state["timestamps"]) >= num_scores else agent.state["timestamps"]
        confidence_scores = agent.state["confidence_scores"][-len(confidence_times):] if len(agent.state["confidence_scores"]) >= len(confidence_times) else agent.state["confidence_scores"]

        plt.plot(confidence_times, confidence_scores, 'purple',
                linewidth=2, marker='o', markersize=4, label='LLM Confidence')
        plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='High Confidence Threshold')
        plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
        plt.xlabel('Time (s)')
        plt.ylabel('Confidence Score')
        plt.title('LLM Analysis Confidence Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

    # Plot 8: Enhanced Alert Timeline
    ax8 = plt.subplot(5, 2, 8)
    innovation_mag = np.sqrt(innovations[:, 0]**2 + innovations[:, 1]**2)
    plt.plot(times, innovation_mag, 'k-', linewidth=1.5, alpha=0.7, label='Innovation Magnitude')

    # Add LLM-enhanced alert markers
    for alert in agent.state["alerts"]:
        color = 'orange' if alert.alert_level == AlertLevel.WARNING else 'red'
        marker = '^' if alert.alert_level == AlertLevel.WARNING else 'v'
        plt.scatter(alert.timestamp, alert.innovation_bias,
                   color=color, marker=marker, s=100,
                   label=f'{alert.alert_level.value.title()} Alert' if alert == agent.state["alerts"][0] or
                   alert.alert_level != agent.state["alerts"][agent.state["alerts"].index(alert)-1].alert_level else "")

    plt.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='Bias Threshold')
    plt.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.xlabel('Time (s)')
    plt.ylabel('Innovation Magnitude (m)')
    plt.title('LLM-Enhanced Alert Timeline')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 9: LLM Analysis Timeline (text summary)
    ax9 = plt.subplot(5, 2, 9)
    ax9.axis('off')

    # Create text summary of LLM insights
    llm_summary = "LLM Analysis Summary:\n\n"
    if agent.state["llm_recommendations"]:
        llm_summary += f"Latest Recommendation:\n{agent.state['llm_recommendations'][-1][:80]}...\n\n"

    llm_summary += f"Total LLM Analyses: {len(agent.state['llm_conversation_history'])}\n"
    llm_summary += f"Avg Confidence: {np.mean(agent.state['confidence_scores']):.2f}\n" if agent.state["confidence_scores"] else "No confidence data\n"
    llm_summary += f"Model Error Detected: {agent.state['model_error_detected']}\n"

    if agent.state["alerts"]:
        llm_summary += f"\nAlert Summary:\n"
        warning_count = sum(1 for alert in agent.state["alerts"] if alert.alert_level == AlertLevel.WARNING)
        critical_count = sum(1 for alert in agent.state["alerts"] if alert.alert_level == AlertLevel.CRITICAL)
        llm_summary += f"  Warnings: {warning_count}\n"
        llm_summary += f"  Critical: {critical_count}\n"

    ax9.text(0.05, 0.95, llm_summary, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax9.set_title('LLM Analysis Summary')

    # Plot 10: Innovation Statistics Over Time
    ax10 = plt.subplot(5, 2, 10)

    # Calculate rolling statistics for visualization
    window_size = 20
    rolling_bias = []
    rolling_variance = []
    rolling_times = []

    for i in range(window_size, len(times)):
        window_innovations_x = innovations[i-window_size:i, 0]
        window_innovations_y = innovations[i-window_size:i, 1]

        bias = np.sqrt(np.mean(window_innovations_x)**2 + np.mean(window_innovations_y)**2)
        variance = np.sqrt(np.var(window_innovations_x) + np.var(window_innovations_y))

        rolling_bias.append(bias)
        rolling_variance.append(variance)
        rolling_times.append(times[i])

    plt.plot(rolling_times, rolling_bias, 'red', linewidth=2, label='Rolling Bias')
    plt.plot(rolling_times, rolling_variance, 'blue', linewidth=2, label='Rolling Variance')
    plt.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='Bias Threshold')
    plt.axvline(x=50, color='k', linestyle=':', alpha=0.7, label='Motion starts')
    plt.xlabel('Time (s)')
    plt.ylabel('Statistical Metric')
    plt.title('Rolling Innovation Statistics')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle('LLM-Enhanced Wrong Model Tracking: AI-Driven Innovation Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save plot
    plot_filename = 'wrong_model_with_llm_tracking.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"LLM-enhanced tracking plots saved to: {plot_filename}")

    print()
    print("=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print("This simulation demonstrates LLM-enhanced model error detection:")
    print("- Target: Static (0-50s) → Constant velocity 10 m/s (50-100s)")
    print("- Tracker: Static model throughout (correct only for first 50s)")
    print("- Agent: Multi-node LangGraph workflow with Ollama LLM integration")
    print()
    print("Key observations:")
    print(f"- During static phase: Good tracking (Pos RMSE = {static_pos_rmse:.3f} m)")
    print(f"- During motion phase: Poor tracking (Pos RMSE = {motion_pos_rmse:.3f} m)")
    print(f"- LLM analyses performed: {len(agent.state['llm_conversation_history'])}")
    print(f"- Enhanced alerts generated: {len(agent.state['alerts'])}")
    print(f"- Average LLM confidence: {np.mean(agent.state['confidence_scores']):.2f}" if agent.state["confidence_scores"] else "- No confidence data available")
    print()
    print("LLM-enhanced agent successfully provided semantic understanding of model errors!")


if __name__ == '__main__':
    main()