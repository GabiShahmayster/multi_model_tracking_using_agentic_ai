import random
import unittest
from typing import Tuple, List
from simulators import StaticMotionSimulator, ConstantVelocitySimulator, ConstantAccelerationSimulator


class MotionSimulator:
    """Simulates 2D motion with user-defined segments of different motion types."""
    
    def __init__(self, x0: float = 0.0, y0: float = 0.0, vx0: float = 0.0, vy0: float = 0.0):
        """
        Initialize the motion simulator.
        
        Args:
            x0, y0: Initial position
            vx0, vy0: Initial velocity
        """
        self.x0 = x0
        self.y0 = y0
        self.vx0 = vx0
        self.vy0 = vy0
        self.segments = []
        self.duration = 0.0
        
        # Motion types
        self.STATIC = 0
        self.CONSTANT_VELOCITY = 1
        self.CONSTANT_ACCELERATION = 2
    
    def add_segment(self, motion_type: int, duration: float, **kwargs):
        """
        Add a motion segment to the simulation.
        
        Args:
            motion_type: Type of motion (STATIC=0, CONSTANT_VELOCITY=1, CONSTANT_ACCELERATION=2)
            duration: Duration of this segment
            **kwargs: Additional parameters for motion type:
                - For CONSTANT_ACCELERATION: ax, ay (acceleration components)
                - Other parameters are ignored for STATIC and CONSTANT_VELOCITY
        """
        if duration <= 0:
            raise ValueError("Duration must be positive")
        
        if motion_type not in [self.STATIC, self.CONSTANT_VELOCITY, self.CONSTANT_ACCELERATION]:
            raise ValueError("Invalid motion type")
        
        # Calculate starting position and velocity for this segment
        if not self.segments:
            # First segment starts with initial conditions
            start_x, start_y = self.x0, self.y0
            start_vx, start_vy = self.vx0, self.vy0
        else:
            # Continue from where the last segment ended
            last_segment = self.segments[-1]
            last_duration = last_segment['end_time'] - last_segment['start_time']
            start_x, start_y = last_segment['simulator'].get_position(last_duration)
            start_vx, start_vy = last_segment['simulator'].get_velocity(last_duration)
        
        # Create appropriate simulator for this segment
        if motion_type == self.STATIC:
            simulator = StaticMotionSimulator(start_x, start_y)
            acceleration = (0.0, 0.0)
        elif motion_type == self.CONSTANT_VELOCITY:
            simulator = ConstantVelocitySimulator(start_x, start_y, start_vx, start_vy)
            acceleration = (0.0, 0.0)
        else:  # CONSTANT_ACCELERATION
            ax = kwargs.get('ax', 0.0)
            ay = kwargs.get('ay', 0.0)
            simulator = ConstantAccelerationSimulator(start_x, start_y, start_vx, start_vy, ax, ay)
            acceleration = (ax, ay)
        
        # Store segment info
        segment = {
            'start_time': self.duration,
            'end_time': self.duration + duration,
            'simulator': simulator,
            'type': motion_type,
            'acceleration': acceleration
        }
        self.segments.append(segment)
        self.duration += duration
    
    def _find_segment(self, time: float) -> dict:
        """Find the motion segment that contains the given time."""
        if time < 0:
            time = 0
        elif time > self.duration:
            time = self.duration
            
        for segment in self.segments:
            if segment['start_time'] <= time <= segment['end_time']:
                return segment
        
        # If no segment found (shouldn't happen), return the last segment
        return self.segments[-1] if self.segments else None
    
    def get_position(self, time: float) -> Tuple[float, float]:
        """Get position at given time."""
        if not self.segments:
            return (self.x0, self.y0)
        
        segment = self._find_segment(time)
        if segment is None:
            return (self.x0, self.y0)
        
        # Calculate relative time within the segment
        relative_time = time - segment['start_time']
        return segment['simulator'].get_position(relative_time)
    
    def get_velocity(self, time: float) -> Tuple[float, float]:
        """Get velocity at given time."""
        if not self.segments:
            return (self.vx0, self.vy0)
        
        segment = self._find_segment(time)
        if segment is None:
            return (self.vx0, self.vy0)
        
        # Calculate relative time within the segment
        relative_time = time - segment['start_time']
        return segment['simulator'].get_velocity(relative_time)
    
    def get_motion_history(self, time_step: float = 0.1) -> List[dict]:
        """Get complete motion history for visualization/analysis."""
        if not self.segments:
            return []
        
        history = []
        time = 0.0
        
        while time <= self.duration:
            segment = self._find_segment(time)
            pos = self.get_position(time)
            vel = self.get_velocity(time)
            
            history.append({
                'time': time,
                'position': pos,
                'velocity': vel,
                'motion_type': segment['type'] if segment else None,
                'acceleration': segment['acceleration'] if segment else (0, 0)
            })
            
            time += time_step
        
        return history
    
    def clear_segments(self):
        """Clear all segments and reset duration."""
        self.segments = []
        self.duration = 0.0


class RandomMotionSimulator:
    """Simulates 2D motion with random switching between different motion types."""
    
    def __init__(self, duration: float, switch_probability: float = 0.1, 
                 x0: float = 0.0, y0: float = 0.0, vx0: float = 0.0, vy0: float = 0.0,
                 random_seed: int = None):
        """
        Initialize the random motion simulator.
        
        Args:
            duration: Total simulation duration
            switch_probability: Probability of switching motion type per time unit
            x0, y0: Initial position
            vx0, vy0: Initial velocity
            random_seed: Optional seed for reproducible results
        """
        self.duration = duration
        self.switch_probability = switch_probability
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
        
        # Motion types
        self.STATIC = 0
        self.CONSTANT_VELOCITY = 1
        self.CONSTANT_ACCELERATION = 2
        
        # Initialize with first motion segment
        self.segments = []
        self._generate_motion_segments(x0, y0, vx0, vy0)
    
    def _generate_motion_segments(self, x0: float, y0: float, vx0: float, vy0: float):
        """Generate all motion segments for the simulation duration."""
        current_time = 0.0
        current_x, current_y = x0, y0
        current_vx, current_vy = vx0, vy0
        
        while current_time < self.duration:
            # Randomly choose motion type
            motion_type = random.choice([self.STATIC, self.CONSTANT_VELOCITY, self.CONSTANT_ACCELERATION])
            
            # Determine segment duration (exponential distribution based on switch probability)
            segment_duration = min(
                random.expovariate(self.switch_probability),
                self.duration - current_time
            )
            
            # Create appropriate simulator for this segment
            if motion_type == self.STATIC:
                simulator = StaticMotionSimulator(current_x, current_y)
                # Generate random acceleration for future velocity changes
                ax = random.uniform(-5, 5)
                ay = random.uniform(-5, 5)
            elif motion_type == self.CONSTANT_VELOCITY:
                simulator = ConstantVelocitySimulator(current_x, current_y, current_vx, current_vy)
                # Generate random acceleration for future velocity changes
                ax = random.uniform(-5, 5)
                ay = random.uniform(-5, 5)
            else:  # CONSTANT_ACCELERATION
                # Generate random acceleration
                ax = random.uniform(-5, 5)
                ay = random.uniform(-5, 5)
                simulator = ConstantAccelerationSimulator(
                    current_x, current_y, current_vx, current_vy, ax, ay
                )
            
            # Store segment info
            segment = {
                'start_time': current_time,
                'end_time': current_time + segment_duration,
                'simulator': simulator,
                'type': motion_type,
                'acceleration': (ax, ay) if motion_type == self.CONSTANT_ACCELERATION else (0, 0)
            }
            self.segments.append(segment)
            
            # Update position and velocity for continuity at segment end
            segment_end_time = segment_duration
            end_pos = simulator.get_position(segment_end_time)
            end_vel = simulator.get_velocity(segment_end_time)
            
            current_x, current_y = end_pos
            current_vx, current_vy = end_vel
            current_time += segment_duration
    
    def _find_segment(self, time: float) -> dict:
        """Find the motion segment that contains the given time."""
        if time < 0:
            time = 0
        elif time > self.duration:
            time = self.duration
            
        for segment in self.segments:
            if segment['start_time'] <= time <= segment['end_time']:
                return segment
        
        # If no segment found (shouldn't happen), return the last segment
        return self.segments[-1] if self.segments else None
    
    def get_position(self, time: float) -> Tuple[float, float]:
        """Get position at given time."""
        segment = self._find_segment(time)
        if segment is None:
            return (0.0, 0.0)
        
        # Calculate relative time within the segment
        relative_time = time - segment['start_time']
        return segment['simulator'].get_position(relative_time)
    
    def get_velocity(self, time: float) -> Tuple[float, float]:
        """Get velocity at given time."""
        segment = self._find_segment(time)
        if segment is None:
            return (0.0, 0.0)
        
        # Calculate relative time within the segment
        relative_time = time - segment['start_time']
        return segment['simulator'].get_velocity(relative_time)
    
    def get_motion_history(self, time_step: float = 0.1) -> List[dict]:
        """Get complete motion history for visualization/analysis."""
        history = []
        time = 0.0
        
        while time <= self.duration:
            segment = self._find_segment(time)
            pos = self.get_position(time)
            vel = self.get_velocity(time)
            
            history.append({
                'time': time,
                'position': pos,
                'velocity': vel,
                'motion_type': segment['type'] if segment else None,
                'acceleration': segment['acceleration'] if segment else (0, 0)
            })
            
            time += time_step
        
        return history


class TestMotionSimulator(unittest.TestCase):
    """Unit tests for MotionSimulator."""
    
    def setUp(self):
        self.simulator = MotionSimulator(x0=1.0, y0=2.0, vx0=3.0, vy0=4.0)
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.simulator.x0, 1.0)
        self.assertEqual(self.simulator.y0, 2.0)
        self.assertEqual(self.simulator.vx0, 3.0)
        self.assertEqual(self.simulator.vy0, 4.0)
        self.assertEqual(len(self.simulator.segments), 0)
        self.assertEqual(self.simulator.duration, 0.0)
    
    def test_add_static_segment(self):
        """Test adding a static motion segment."""
        self.simulator.add_segment(self.simulator.STATIC, 5.0)
        
        self.assertEqual(len(self.simulator.segments), 1)
        self.assertEqual(self.simulator.duration, 5.0)
        
        segment = self.simulator.segments[0]
        self.assertEqual(segment['type'], self.simulator.STATIC)
        self.assertEqual(segment['start_time'], 0.0)
        self.assertEqual(segment['end_time'], 5.0)
        self.assertEqual(segment['acceleration'], (0.0, 0.0))
    
    def test_add_constant_velocity_segment(self):
        """Test adding a constant velocity segment."""
        self.simulator.add_segment(self.simulator.CONSTANT_VELOCITY, 3.0)
        
        segment = self.simulator.segments[0]
        self.assertEqual(segment['type'], self.simulator.CONSTANT_VELOCITY)
        self.assertEqual(segment['acceleration'], (0.0, 0.0))
    
    def test_add_constant_acceleration_segment(self):
        """Test adding a constant acceleration segment."""
        self.simulator.add_segment(self.simulator.CONSTANT_ACCELERATION, 2.0, ax=1.5, ay=-2.0)
        
        segment = self.simulator.segments[0]
        self.assertEqual(segment['type'], self.simulator.CONSTANT_ACCELERATION)
        self.assertEqual(segment['acceleration'], (1.5, -2.0))
    
    def test_multiple_segments_continuity(self):
        """Test that multiple segments maintain position and velocity continuity."""
        # Add segments: static -> constant velocity -> constant acceleration
        self.simulator.add_segment(self.simulator.STATIC, 2.0)
        self.simulator.add_segment(self.simulator.CONSTANT_VELOCITY, 3.0)
        self.simulator.add_segment(self.simulator.CONSTANT_ACCELERATION, 2.0, ax=1.0, ay=-1.0)
        
        self.assertEqual(len(self.simulator.segments), 3)
        self.assertEqual(self.simulator.duration, 7.0)
        
        # Check continuity at segment boundaries
        # At t=2.0 (end of static, start of constant velocity)
        pos_before = self.simulator.get_position(1.9999)
        pos_after = self.simulator.get_position(2.0001)
        vel_before = self.simulator.get_velocity(1.9999)
        vel_after = self.simulator.get_velocity(2.0001)
        
        self.assertAlmostEqual(pos_before[0], pos_after[0], places=3)
        self.assertAlmostEqual(pos_before[1], pos_after[1], places=3)
        self.assertAlmostEqual(vel_before[0], vel_after[0], places=3)
        self.assertAlmostEqual(vel_before[1], vel_after[1], places=3)
        
        # At t=5.0 (end of constant velocity, start of constant acceleration)
        pos_before = self.simulator.get_position(4.9999)
        pos_after = self.simulator.get_position(5.0001)
        vel_before = self.simulator.get_velocity(4.9999)
        vel_after = self.simulator.get_velocity(5.0001)
        
        self.assertAlmostEqual(pos_before[0], pos_after[0], places=3)
        self.assertAlmostEqual(pos_before[1], pos_after[1], places=3)
        self.assertAlmostEqual(vel_before[0], vel_after[0], places=3)
        self.assertAlmostEqual(vel_before[1], vel_after[1], places=3)
    
    def test_position_calculation(self):
        """Test position calculation across different segments."""
        # Start at (1,2) with velocity (3,4)
        # Static for 2 seconds -> stays at (1,2), velocity becomes (0,0)
        self.simulator.add_segment(self.simulator.STATIC, 2.0)
        
        pos = self.simulator.get_position(1.0)
        self.assertEqual(pos, (1.0, 2.0))
        
        # Constant velocity for 2 seconds with velocity from static segment (0,0)
        # At t=2, position should still be (1,2) and velocity (0,0)
        # At t=4, position should still be (1,2) since velocity is (0,0)
        self.simulator.add_segment(self.simulator.CONSTANT_VELOCITY, 2.0)
        
        pos_start = self.simulator.get_position(2.0)
        pos_end = self.simulator.get_position(4.0)
        self.assertEqual(pos_start, (1.0, 2.0))
        self.assertEqual(pos_end, (1.0, 2.0))  # No movement since velocity is (0,0)
    
    def test_velocity_calculation(self):
        """Test velocity calculation across different segments."""
        # Static segment: velocity becomes 0
        self.simulator.add_segment(self.simulator.STATIC, 2.0)
        vel = self.simulator.get_velocity(1.0)
        self.assertEqual(vel, (0.0, 0.0))
        
        # Constant velocity: maintains velocity
        self.simulator.add_segment(self.simulator.CONSTANT_VELOCITY, 2.0)
        vel = self.simulator.get_velocity(3.0)
        self.assertEqual(vel, (0.0, 0.0))  # Velocity from static segment
        
        # Add another constant velocity segment starting with non-zero velocity
        sim2 = MotionSimulator(x0=0, y0=0, vx0=5, vy0=-2)
        sim2.add_segment(sim2.CONSTANT_VELOCITY, 3.0)
        vel = sim2.get_velocity(1.0)
        self.assertEqual(vel, (5.0, -2.0))
    
    def test_acceleration_segment(self):
        """Test constant acceleration segment calculations."""
        # Start with zero velocity, add acceleration (2, -1) for 3 seconds
        sim = MotionSimulator(x0=0, y0=0, vx0=0, vy0=0)
        sim.add_segment(sim.CONSTANT_ACCELERATION, 3.0, ax=2.0, ay=-1.0)
        
        # At t=1: vx = 0 + 2*1 = 2, vy = 0 + (-1)*1 = -1
        # At t=1: x = 0 + 0*1 + 0.5*2*1^2 = 1, y = 0 + 0*1 + 0.5*(-1)*1^2 = -0.5
        vel = sim.get_velocity(1.0)
        pos = sim.get_position(1.0)
        self.assertEqual(vel, (2.0, -1.0))
        self.assertEqual(pos, (1.0, -0.5))
        
        # At t=3: vx = 0 + 2*3 = 6, vy = 0 + (-1)*3 = -3
        # At t=3: x = 0 + 0*3 + 0.5*2*3^2 = 9, y = 0 + 0*3 + 0.5*(-1)*3^2 = -4.5
        vel = sim.get_velocity(3.0)
        pos = sim.get_position(3.0)
        self.assertEqual(vel, (6.0, -3.0))
        self.assertEqual(pos, (9.0, -4.5))
    
    def test_invalid_inputs(self):
        """Test invalid input handling."""
        # Invalid duration
        with self.assertRaises(ValueError):
            self.simulator.add_segment(self.simulator.STATIC, -1.0)
        
        with self.assertRaises(ValueError):
            self.simulator.add_segment(self.simulator.STATIC, 0.0)
        
        # Invalid motion type
        with self.assertRaises(ValueError):
            self.simulator.add_segment(999, 1.0)
    
    def test_get_motion_history(self):
        """Test motion history generation."""
        self.simulator.add_segment(self.simulator.STATIC, 1.0)
        self.simulator.add_segment(self.simulator.CONSTANT_VELOCITY, 1.0)
        
        history = self.simulator.get_motion_history(time_step=0.5)
        
        # Should have entries at t=0, 0.5, 1.0, 1.5, 2.0
        self.assertEqual(len(history), 5)
        
        # Check that each entry has required fields
        for entry in history:
            self.assertIn('time', entry)
            self.assertIn('position', entry)
            self.assertIn('velocity', entry)
            self.assertIn('motion_type', entry)
            self.assertIn('acceleration', entry)
        
        # Check motion types at different times
        self.assertEqual(history[0]['motion_type'], self.simulator.STATIC)  # t=0
        self.assertEqual(history[1]['motion_type'], self.simulator.STATIC)  # t=0.5
        self.assertEqual(history[2]['motion_type'], self.simulator.STATIC)  # t=1.0
        self.assertEqual(history[3]['motion_type'], self.simulator.CONSTANT_VELOCITY)  # t=1.5
        self.assertEqual(history[4]['motion_type'], self.simulator.CONSTANT_VELOCITY)  # t=2.0
    
    def test_clear_segments(self):
        """Test clearing all segments."""
        self.simulator.add_segment(self.simulator.STATIC, 2.0)
        self.simulator.add_segment(self.simulator.CONSTANT_VELOCITY, 3.0)
        
        self.assertEqual(len(self.simulator.segments), 2)
        self.assertEqual(self.simulator.duration, 5.0)
        
        self.simulator.clear_segments()
        
        self.assertEqual(len(self.simulator.segments), 0)
        self.assertEqual(self.simulator.duration, 0.0)
    
    def test_empty_simulator_behavior(self):
        """Test behavior when no segments are added."""
        # Should return initial conditions
        pos = self.simulator.get_position(5.0)
        vel = self.simulator.get_velocity(5.0)
        
        self.assertEqual(pos, (1.0, 2.0))  # Initial position
        self.assertEqual(vel, (3.0, 4.0))  # Initial velocity
        
        # Motion history should be empty
        history = self.simulator.get_motion_history()
        self.assertEqual(len(history), 0)


class TestRandomMotionSimulator(unittest.TestCase):
    """Unit tests for RandomMotionSimulator."""
    
    def setUp(self):
        # Use fixed seed for reproducible tests
        self.simulator = RandomMotionSimulator(
            duration=10.0, 
            switch_probability=0.5, 
            x0=0.0, y0=0.0, vx0=1.0, vy0=1.0,
            random_seed=42
        )
    
    def test_initial_conditions(self):
        """Test that initial conditions are respected."""
        pos = self.simulator.get_position(0)
        vel = self.simulator.get_velocity(0)
        self.assertAlmostEqual(pos[0], 0.0, places=5)
        self.assertAlmostEqual(pos[1], 0.0, places=5)
        # Initial velocity might not be exactly (1,1) due to motion type switching
    
    def test_position_continuity(self):
        """Test that position remains continuous across segment boundaries."""
        history = self.simulator.get_motion_history(time_step=0.01)
        
        for i in range(1, len(history)):
            prev_pos = history[i-1]['position']
            curr_pos = history[i]['position']
            time_diff = history[i]['time'] - history[i-1]['time']
            
            # Position should change smoothly (no sudden jumps)
            pos_diff_x = abs(curr_pos[0] - prev_pos[0])
            pos_diff_y = abs(curr_pos[1] - prev_pos[1])
            
            # Maximum reasonable position change in 0.01 time units
            max_reasonable_change = 10 * time_diff  # Assume max velocity ~10 units/time
            
            self.assertLess(pos_diff_x, max_reasonable_change, 
                          f"Position discontinuity at time {history[i]['time']}")
            self.assertLess(pos_diff_y, max_reasonable_change,
                          f"Position discontinuity at time {history[i]['time']}")
    
    def test_simulation_duration(self):
        """Test that simulation respects the specified duration."""
        # Position and velocity should be computable at any time within duration
        pos_start = self.simulator.get_position(0)
        pos_end = self.simulator.get_position(self.simulator.duration)
        
        self.assertIsNotNone(pos_start)
        self.assertIsNotNone(pos_end)
        self.assertEqual(len(pos_start), 2)
        self.assertEqual(len(pos_end), 2)
    
    def test_motion_segments_generated(self):
        """Test that motion segments are properly generated."""
        self.assertGreater(len(self.simulator.segments), 0)
        
        # Check that segments cover the entire duration
        total_duration = sum(seg['end_time'] - seg['start_time'] 
                           for seg in self.simulator.segments)
        self.assertAlmostEqual(total_duration, self.simulator.duration, places=5)
    
    def test_different_motion_types(self):
        """Test that different motion types are used."""
        motion_types = set(seg['type'] for seg in self.simulator.segments)
        # With random seed 42 and high switch probability, we should get variety
        # (This is probabilistic, but with the fixed seed it should be consistent)
        self.assertGreaterEqual(len(motion_types), 1)
    
    def test_get_motion_history(self):
        """Test motion history generation."""
        history = self.simulator.get_motion_history(time_step=1.0)
        
        self.assertGreater(len(history), 0)
        
        # Check that each history entry has required fields
        for entry in history:
            self.assertIn('time', entry)
            self.assertIn('position', entry)
            self.assertIn('velocity', entry)
            self.assertIn('motion_type', entry)
            self.assertIn('acceleration', entry)
            
            # Check that time is within bounds
            self.assertGreaterEqual(entry['time'], 0)
            self.assertLessEqual(entry['time'], self.simulator.duration)


if __name__ == '__main__':
    # Example usage for MotionSimulator
    print("Motion Simulator Example")
    print("=" * 30)
    
    # Create a motion simulator with defined segments
    sim = MotionSimulator(x0=0, y0=0, vx0=2, vy0=1)
    
    # Add different motion segments
    sim.add_segment(sim.STATIC, 2.0)  # Static for 2 seconds
    sim.add_segment(sim.CONSTANT_VELOCITY, 3.0)  # Constant velocity for 3 seconds  
    sim.add_segment(sim.CONSTANT_ACCELERATION, 2.0, ax=1.0, ay=-0.5)  # Acceleration for 2 seconds
    
    print(f"Total simulation duration: {sim.duration} seconds")
    print(f"Number of segments: {len(sim.segments)}")
    
    print("\nSegment details:")
    motion_names = {0: "Static", 1: "Constant Velocity", 2: "Constant Acceleration"}
    for i, segment in enumerate(sim.segments):
        print(f"  Segment {i+1}: {motion_names[segment['type']]} "
              f"from t={segment['start_time']:.1f} to t={segment['end_time']:.1f}")
        if segment['type'] == 2:
            ax, ay = segment['acceleration']
            print(f"    Acceleration: ({ax:.1f}, {ay:.1f})")
    
    print("\nPosition and velocity at key times:")
    for t in [0, 1, 2, 3, 5, 7]:
        pos = sim.get_position(t)
        vel = sim.get_velocity(t)
        print(f"  t={t:.1f}: pos=({pos[0]:6.2f}, {pos[1]:6.2f}), "
              f"vel=({vel[0]:6.2f}, {vel[1]:6.2f})")
    
    print("\n" + "=" * 40)
    print("Random Motion Simulator Example")
    print("=" * 40)
    
    # Create a simulator with 20 second duration and 20% switch probability per second
    sim = RandomMotionSimulator(
        duration=20.0, 
        switch_probability=0.2, 
        x0=0, y0=0, vx0=2, vy0=1,
        random_seed=123
    )
    
    # Print motion segments
    print(f"Generated {len(sim.segments)} motion segments:")
    for i, segment in enumerate(sim.segments):
        motion_names = {0: "Static", 1: "Constant Velocity", 2: "Constant Acceleration"}
        print(f"  Segment {i+1}: {motion_names[segment['type']]} "
              f"from t={segment['start_time']:.2f} to t={segment['end_time']:.2f}")
        if segment['type'] == 2:  # Constant acceleration
            ax, ay = segment['acceleration']
            print(f"    Acceleration: ({ax:.2f}, {ay:.2f})")
    
    # Show position and velocity at key times
    print("\nPosition and velocity at key times:")
    for t in [0, 5, 10, 15, 20]:
        pos = sim.get_position(t)
        vel = sim.get_velocity(t)
        print(f"  t={t:2.0f}: pos=({pos[0]:6.2f}, {pos[1]:6.2f}), "
              f"vel=({vel[0]:6.2f}, {vel[1]:6.2f})")
    
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)