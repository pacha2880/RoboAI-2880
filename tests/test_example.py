"""
Example tests for the robot navigation challenge.

Candidates are encouraged to add more tests!
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from robot_simulator import DifferentialDriveRobot, RobotParams


class TestDifferentialDriveRobot:
    """Tests for the robot simulator."""
    
    def test_robot_initialization(self):
        """Test that robot initializes correctly."""
        robot = DifferentialDriveRobot()
        state = robot.get_state()
        
        assert len(state) == 3
        assert state[0] == 0.0  # x
        assert state[1] == 0.0  # y
        assert state[2] == 0.0  # theta
    
    def test_robot_reset(self):
        """Test that robot reset works."""
        robot = DifferentialDriveRobot()
        robot.reset(1.0, 2.0, np.pi/4)
        state = robot.get_state()
        
        assert np.isclose(state[0], 1.0)
        assert np.isclose(state[1], 2.0)
        assert np.isclose(state[2], np.pi/4)
    
    def test_straight_line_motion_ideal(self):
        """Test that ideal robot goes straight when given equal wheel speeds."""
        robot = DifferentialDriveRobot(realistic=False)
        robot.reset(0, 0, 0)  # Facing +x direction
        
        # Apply equal wheel velocities
        for _ in range(10):
            robot.step(0.05, 0.05)
        
        state = robot.get_state()
        
        # Should have moved forward in x, no movement in y
        assert state[0] > 0  # Moved forward
        assert np.abs(state[1]) < 1e-6  # No lateral movement
        assert np.abs(state[2]) < 1e-6  # No rotation
    
    def test_turning_motion(self):
        """Test that robot turns when wheels have different speeds."""
        robot = DifferentialDriveRobot(realistic=False)
        robot.reset(0, 0, 0)
        
        # Right wheel faster -> should turn left (positive theta)
        for _ in range(10):
            robot.step(0.0, 0.05)
        
        state = robot.get_state()
        assert state[2] > 0  # Should have turned left
    
    def test_realistic_vs_ideal_difference(self):
        """Test that realistic robot behaves differently from ideal."""
        robot_ideal = DifferentialDriveRobot(realistic=False)
        robot_real = DifferentialDriveRobot(realistic=True)
        
        controls = [[0.05, 0.05]] * 50
        
        robot_ideal.reset(0, 0, 0)
        robot_real.reset(0, 0, 0)
        
        states_ideal = robot_ideal.simulate_trajectory(np.array(controls))
        states_real = robot_real.simulate_trajectory(np.array(controls))
        
        # Final positions should be different
        final_diff = np.linalg.norm(states_ideal[-1][:2] - states_real[-1][:2])
        
        # This test reveals that there IS a difference!
        # Students should investigate WHY
        assert final_diff > 0.01, "Expected realistic robot to deviate from ideal"


class TestDataset:
    """Tests for dataset generation."""
    
    def test_dataset_generation(self):
        """Test basic dataset generation."""
        from dataset import generate_dataset
        
        dataset = generate_dataset(n_trajectories=5, seed=42)
        
        assert 'inputs' in dataset
        assert 'outputs' in dataset
        assert len(dataset['inputs']) > 0
        assert len(dataset['inputs']) == len(dataset['outputs'])
    
    def test_input_output_shapes(self):
        """Test that inputs and outputs have correct shapes."""
        from dataset import generate_dataset
        
        dataset = generate_dataset(n_trajectories=5, seed=42)
        
        # Input: [x, y, theta, x_target, y_target]
        assert dataset['inputs'].shape[1] == 5
        
        # Output: [v_left, v_right]
        assert dataset['outputs'].shape[1] == 2


class TestModel:
    """Tests for the neural network model."""
    
    def test_model_forward_pass(self):
        """Test that model produces correct output shape."""
        import torch
        from model import NavigationNet
        
        model = NavigationNet()
        x = torch.randn(10, 5)  # Batch of 10 samples
        y = model(x)
        
        assert y.shape == (10, 2)
    
    def test_model_output_range(self):
        """Test model output range (reveals tanh limitation)."""
        import torch
        from model import NavigationNet
        
        model = NavigationNet()
        x = torch.randn(100, 5)
        y = model(x)
        
        # Due to tanh, outputs should be in [-1, 1]
        # Students might question if this is appropriate!
        assert y.min() >= -1.0
        assert y.max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
