import unittest
import math
from typing import Tuple


class StaticMotionSimulator:
    """Simulates 2D static motion - object remains at fixed position."""
    
    def __init__(self, x: float = 0.0, y: float = 0.0):
        """Initialize with starting position."""
        self.x = x
        self.y = y
    
    def get_position(self, time: float) -> Tuple[float, float]:
        """Get position at given time. For static motion, position never changes."""
        return (self.x, self.y)
    
    def get_velocity(self, time: float) -> Tuple[float, float]:
        """Get velocity at given time. For static motion, velocity is always zero."""
        return (0.0, 0.0)


class ConstantVelocitySimulator:
    """Simulates 2D constant velocity motion."""
    
    def __init__(self, x0: float = 0.0, y0: float = 0.0, vx: float = 0.0, vy: float = 0.0):
        """Initialize with starting position and constant velocity."""
        self.x0 = x0
        self.y0 = y0
        self.vx = vx
        self.vy = vy
    
    def get_position(self, time: float) -> Tuple[float, float]:
        """Get position at given time using x = x0 + vx*t, y = y0 + vy*t."""
        x = self.x0 + self.vx * time
        y = self.y0 + self.vy * time
        return (x, y)
    
    def get_velocity(self, time: float) -> Tuple[float, float]:
        """Get velocity at given time. For constant velocity, velocity never changes."""
        return (self.vx, self.vy)


class ConstantAccelerationSimulator:
    """Simulates 2D constant acceleration motion."""
    
    def __init__(self, x0: float = 0.0, y0: float = 0.0, vx0: float = 0.0, vy0: float = 0.0, 
                 ax: float = 0.0, ay: float = 0.0):
        """Initialize with starting position, initial velocity, and constant acceleration."""
        self.x0 = x0
        self.y0 = y0
        self.vx0 = vx0
        self.vy0 = vy0
        self.ax = ax
        self.ay = ay
    
    def get_position(self, time: float) -> Tuple[float, float]:
        """Get position at given time using x = x0 + vx0*t + 0.5*ax*t^2."""
        x = self.x0 + self.vx0 * time + 0.5 * self.ax * time * time
        y = self.y0 + self.vy0 * time + 0.5 * self.ay * time * time
        return (x, y)
    
    def get_velocity(self, time: float) -> Tuple[float, float]:
        """Get velocity at given time using vx = vx0 + ax*t."""
        vx = self.vx0 + self.ax * time
        vy = self.vy0 + self.ay * time
        return (vx, vy)


class TestStaticMotionSimulator(unittest.TestCase):
    """Unit tests for StaticMotionSimulator."""
    
    def setUp(self):
        self.simulator = StaticMotionSimulator(5.0, 3.0)
    
    def test_position_remains_constant(self):
        """Test that position remains constant over time."""
        for t in [0, 1, 5, 10, 100]:
            pos = self.simulator.get_position(t)
            self.assertEqual(pos, (5.0, 3.0))
    
    def test_velocity_is_always_zero(self):
        """Test that velocity is always zero."""
        for t in [0, 1, 5, 10, 100]:
            vel = self.simulator.get_velocity(t)
            self.assertEqual(vel, (0.0, 0.0))
    
    def test_default_initialization(self):
        """Test default initialization at origin."""
        sim = StaticMotionSimulator()
        self.assertEqual(sim.get_position(0), (0.0, 0.0))
        self.assertEqual(sim.get_velocity(0), (0.0, 0.0))


class TestConstantVelocitySimulator(unittest.TestCase):
    """Unit tests for ConstantVelocitySimulator."""
    
    def setUp(self):
        self.simulator = ConstantVelocitySimulator(x0=2.0, y0=1.0, vx=3.0, vy=-2.0)
    
    def test_position_calculation(self):
        """Test position calculation at various times."""
        self.assertEqual(self.simulator.get_position(0), (2.0, 1.0))
        self.assertEqual(self.simulator.get_position(1), (5.0, -1.0))
        self.assertEqual(self.simulator.get_position(2), (8.0, -3.0))
        self.assertEqual(self.simulator.get_position(0.5), (3.5, 0.0))
    
    def test_velocity_remains_constant(self):
        """Test that velocity remains constant over time."""
        for t in [0, 1, 5, 10, 100]:
            vel = self.simulator.get_velocity(t)
            self.assertEqual(vel, (3.0, -2.0))
    
    def test_default_initialization(self):
        """Test default initialization with zero velocity."""
        sim = ConstantVelocitySimulator()
        self.assertEqual(sim.get_position(5), (0.0, 0.0))
        self.assertEqual(sim.get_velocity(5), (0.0, 0.0))


class TestConstantAccelerationSimulator(unittest.TestCase):
    """Unit tests for ConstantAccelerationSimulator."""
    
    def setUp(self):
        self.simulator = ConstantAccelerationSimulator(
            x0=1.0, y0=2.0, vx0=3.0, vy0=4.0, ax=2.0, ay=-1.0
        )
    
    def test_position_calculation(self):
        """Test position calculation at various times."""
        # At t=0: x = 1, y = 2
        self.assertEqual(self.simulator.get_position(0), (1.0, 2.0))
        
        # At t=1: x = 1 + 3*1 + 0.5*2*1^2 = 5, y = 2 + 4*1 + 0.5*(-1)*1^2 = 5.5
        pos = self.simulator.get_position(1)
        self.assertAlmostEqual(pos[0], 5.0)
        self.assertAlmostEqual(pos[1], 5.5)
        
        # At t=2: x = 1 + 3*2 + 0.5*2*2^2 = 11, y = 2 + 4*2 + 0.5*(-1)*2^2 = 8
        pos = self.simulator.get_position(2)
        self.assertAlmostEqual(pos[0], 11.0)
        self.assertAlmostEqual(pos[1], 8.0)
    
    def test_velocity_calculation(self):
        """Test velocity calculation at various times."""
        # At t=0: vx = 3, vy = 4
        self.assertEqual(self.simulator.get_velocity(0), (3.0, 4.0))
        
        # At t=1: vx = 3 + 2*1 = 5, vy = 4 + (-1)*1 = 3
        self.assertEqual(self.simulator.get_velocity(1), (5.0, 3.0))
        
        # At t=2: vx = 3 + 2*2 = 7, vy = 4 + (-1)*2 = 2
        self.assertEqual(self.simulator.get_velocity(2), (7.0, 2.0))
    
    def test_default_initialization(self):
        """Test default initialization with zero values."""
        sim = ConstantAccelerationSimulator()
        self.assertEqual(sim.get_position(0), (0.0, 0.0))
        self.assertEqual(sim.get_velocity(0), (0.0, 0.0))
    
    def test_free_fall_simulation(self):
        """Test simulation of free fall (constant downward acceleration)."""
        # Object starts at height 10m with zero velocity, gravity = -9.8 m/s^2
        sim = ConstantAccelerationSimulator(x0=0, y0=10, vx0=0, vy0=0, ax=0, ay=-9.8)
        
        # At t=1s: y = 10 + 0*1 + 0.5*(-9.8)*1^2 = 5.1
        pos = sim.get_position(1)
        self.assertAlmostEqual(pos[1], 5.1)
        
        # At t=1s: vy = 0 + (-9.8)*1 = -9.8
        vel = sim.get_velocity(1)
        self.assertAlmostEqual(vel[1], -9.8)


if __name__ == '__main__':
    unittest.main()