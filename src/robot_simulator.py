"""
Differential drive robot simulator for 2D navigation.

State: [x, y, theta] where theta is heading in radians.
Control: [v_left, v_right] wheel velocities.
"""

from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tomli

from utils import ROBOT_FIGURES_PATH, THRESHOLD


@dataclass
class RobotParams:
    """Robot physical parameters: wheel_radius, wheel_base, max_wheel_speed."""

    def __init__(self, wheel_radius: float, wheel_base: float, max_wheel_speed: float):
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.max_wheel_speed = max_wheel_speed
        self.max_v = wheel_radius * max_wheel_speed


class DifferentialDriveRobot:
    """Two-wheeled differential drive robot with optional motor imperfections."""

    def __init__(self, params: RobotParams, realistic: bool):
        """
        Args:
            params: Robot physical parameters
            realistic: If True, applies motor calibration factors (left motor 15% weaker)
        """
        self.params = params
        self.realistic = realistic

        # Initial state: [x, y, theta]
        self.state = np.array([0.0, 0.0, 0.0])

        # Motor calibration factors
        # In a perfect robot, these would both be 1.0
        if realistic:
            # Manufacturing variation in motors use less than 1.0 for the weaker motor
            self._left_motor_factor = 1.0 / 1.15
            self._right_motor_factor = 1.0
        else:
            self._left_motor_factor = 1.0
            self._right_motor_factor = 1.0

    def reset_state(self, x: float, y: float, theta: float):
        """Set robot position and orientation."""
        self.state = np.array([x, y, theta])

    def get_state(self) -> np.ndarray:
        """Return current state [x, y, theta]."""
        return self.state.copy()

    def step(self, v_left: float, v_right: float, dt: float) -> np.ndarray:
        """
        Update robot state using wheel velocities and Euler integration.

        Args:
            v_left: Left wheel velocity
            v_right: Right wheel velocity
            dt: Time step

        Returns:
            New state [x, y, theta]
        """
        # Clip velocities to physical limits
        max_v = self.params.max_v
        v_left = np.clip(v_left, -max_v, max_v)
        v_right = np.clip(v_right, -max_v, max_v)

        # Apply motor calibration factors
        v_left_actual = v_left * self._left_motor_factor
        v_right_actual = v_right * self._right_motor_factor

        # Differential drive kinematics
        # Linear and angular velocity of the robot
        v = (v_left_actual + v_right_actual) / 2.0
        omega = (v_right_actual - v_left_actual) / self.params.wheel_base

        # Current state
        x, y, theta = self.state

        # Update state using Euler integration
        if abs(omega) < 1e-6:
            # Approximately straight motion
            x_new = x + v * np.cos(theta) * dt
            y_new = y + v * np.sin(theta) * dt
            theta_new = theta
        else:
            # Curved motion
            x_new = x + v * np.cos(theta) * dt
            y_new = y + v * np.sin(theta) * dt
            theta_new = theta + omega * dt

        self.state = np.array([x_new, y_new, theta_new])
        return self.state.copy()

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def simulate_trajectory(self, controls: np.ndarray, dt: float) -> np.ndarray:
        """
        Simulate trajectory from sequence of controls.

        Args:
            controls: Array (T, 2) of [v_left, v_right] per timestep
            dt: Time step

        Returns:
            states: Array (T+1, 3) of [x, y, theta] including initial state
        """

        states = [self.get_state()]
        for v_left, v_right in controls:
            state = self.step(v_left, v_right, dt)
            states.append(state)

        return np.array(states)


class ExpertController:
    """Polar coordinate controller for generating expert trajectories."""

    def __init__(self, params: RobotParams, robot: DifferentialDriveRobot):
        """Initialize controller with robot parameters and instance."""
        self.params = params
        self.robot = robot
        self.max_v = self.params.max_v

        # Control gains
        self.k_rho = 1.0  # gain for distance
        self.k_alpha = 3.0  # gain for heading to target
        self.k_beta = -0.5  # gain for final orientation

    def compute_control(self, current_state: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
        """
        Compute wheel velocities using polar coordinate controller.

        Args:
            current_state: [x, y, theta]
            target: [x_target, y_target] or [x_target, y_target, theta_target]

        Returns:
            (v_left, v_right) wheel velocities
        """
        x, y, theta = current_state
        x_t, y_t = target[0], target[1]
        theta_t = target[2] if len(target) > 2 else 0.0

        # Transform to polar coordinates relative to target
        dx = x_t - x
        dy = y_t - y

        rho = np.sqrt(dx**2 + dy**2)  # distance to target

        # Angle to target from robot's perspective
        alpha = np.arctan2(dy, dx) - theta
        alpha = self._normalize_angle(alpha)

        # Desired final orientation difference
        beta = theta_t - theta - alpha
        beta = self._normalize_angle(beta)

        # Control law
        if rho < THRESHOLD:  # Close enough to target
            v = 0.0
            omega = 0.0
        else:
            # Standard polar coordinate controller: v includes cos(alpha) to prevent
            # moving forward when not facing the target
            v = self.k_rho * rho * np.cos(alpha)
            omega = self.k_alpha * alpha + self.k_beta * beta

        # Convert to wheel velocities
        v_left = v - (omega * self.params.wheel_base / 2)
        v_right = v + (omega * self.params.wheel_base / 2)

        # Clip to physical limits
        v_left = np.clip(v_left, -self.max_v, self.max_v)
        v_right = np.clip(v_right, -self.max_v, self.max_v)

        return v_left, v_right

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def generate_trajectory(
        self, start: np.ndarray, target: np.ndarray, max_steps: int, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trajectory from start to target using expert controller.

        Args:
            start: [x, y, theta] initial state
            target: [x_target, y_target] or [x_target, y_target, theta_target]
            max_steps: Maximum simulation steps
            dt: Time step

        Returns:
            (states, controls): states (T+1, 3), controls (T, 2)
        """
        # Expert generates data using idealized robot
        robot = self.robot
        robot.reset_state(start[0], start[1], start[2])

        states = [robot.get_state()]
        controls = []

        for _ in range(max_steps):
            state = robot.get_state()

            # Check if reached target
            dist = compute_distance_to_target(state, target)
            if dist < THRESHOLD:
                break

            # Compute and apply control
            v_left, v_right = self.compute_control(state, target)
            controls.append([v_left, v_right])

            new_state = robot.step(v_left, v_right, dt)
            states.append(new_state)

        return np.array(states), np.array(controls)


def compute_distance_to_target(state: np.ndarray, target: np.ndarray) -> float:
    """Return Euclidean distance from robot state to target position."""
    return np.sqrt((state[0] - target[0]) ** 2 + (state[1] - target[1]) ** 2)


def visualize_trajectory(
    states: np.ndarray, target: np.ndarray, title: str = "Robot Trajectory", ax=None
):
    """
    Plot robot trajectory with orientation arrows.

    Args:
        states: Array (T, 3) of [x, y, theta] per timestep
        target: [x_target, y_target]
        title: Plot title
        ax: Optional matplotlib axis (creates new figure if None)

    Returns:
        matplotlib axis
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Plot trajectory
    ax.plot(states[:, 0], states[:, 1], "b-", linewidth=2, label="Trajectory")
    ax.plot(states[0, 0], states[0, 1], "go", markersize=10, label="Start")
    ax.plot(states[-1, 0], states[-1, 1], "rs", markersize=10, label="End")
    ax.plot(target[0], target[1], "r*", markersize=15, label="Target")

    # Plot robot orientation arrows at intervals
    step = max(1, len(states) // 10)
    for i in range(0, len(states), step):
        x, y, theta = states[i]
        dx = 0.1 * np.cos(theta)
        dy = 0.1 * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=0.03, head_length=0.02, fc="blue", ec="blue")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend()
    ax.axis("equal")
    ax.grid(True)

    return ax

def parse_config_robot(config_dict: dict) -> dict:
    """Parse robot simulator configuration from TOML config dictionary."""
    params_dict = config_dict["robot_simulator"]["params"]
    params = RobotParams(
        wheel_radius=params_dict["wheel_radius"],
        wheel_base=params_dict["wheel_base"],
        max_wheel_speed=params_dict["max_wheel_speed"],
    )
    dt = config_dict["robot_simulator"]["euler_integration"]["dt"]
    max_steps = config_dict["robot_simulator"]["euler_integration"]["max_steps"]
    return params, dt, max_steps

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/robo_simulator_default.toml")
    args = parser.parse_args()

    # parse args from toml config file
    with open(args.config, "rb") as f:
        config = tomli.load(f)

    params, dt, max_steps = parse_config_robot(config)

    # Quick test of the simulator
    print("Testing robot simulator...")

    # Test with idealized robot
    robot_ideal = DifferentialDriveRobot(params=params, realistic=False)
    robot_ideal.reset_state(0, 0, 0)

    # Test with realistic robot
    robot_real = DifferentialDriveRobot(params=params, realistic=True)
    robot_real.reset_state(0, 0, 0)

    # Apply same controls to both
    controls = [[0.1, 0.1]] * 50  # Try to go straight

    states_ideal = robot_ideal.simulate_trajectory(np.array(controls), dt)
    states_real = robot_real.simulate_trajectory(np.array(controls), dt)

    print(f"Ideal robot final position: {states_ideal[-1][:2]}")
    print(f"Real robot final position: {states_real[-1][:2]}")

    # Test with expert controller
    expert_real = ExpertController(params=params, robot=robot_real)
    expert_ideal = ExpertController(params=params, robot=robot_ideal)
    target = np.array([0.0, -1.0])
    # target = np.array([1.0, 1.0])
    initial_state = np.array([0.0, 0.0, 0.2])

    # With realistic robot
    robot_real.reset_state(initial_state[0], initial_state[1], initial_state[2])
    states_expert_real, controls_expert_real = expert_real.generate_trajectory(
        robot_real.get_state(), target, max_steps, dt
    )

    # With ideal robot
    robot_ideal.reset_state(0, 0, 0.2)
    states_expert_ideal, controls_expert_ideal = expert_ideal.generate_trajectory(
        robot_ideal.get_state(), target, max_steps, dt
    )

    # Visualize trajectories
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))
    visualize_trajectory(
        states_expert_real, target, title="Expert Robot Trajectory with Realistic Robot", ax=axes[0]
    )
    visualize_trajectory(
        states_expert_ideal, target, title="Expert Robot Trajectory with Ideal Robot", ax=axes[1]
    )
    plt.tight_layout()
    plt.savefig(str(ROBOT_FIGURES_PATH / "expert_trajectories_example.png"))
    print(
        f"Expert trajectories example saved to {ROBOT_FIGURES_PATH / 'expert_trajectories_example.png'}"
    )


if __name__ == "__main__":
    main()
