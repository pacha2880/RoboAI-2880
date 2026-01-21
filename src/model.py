"""
Neural Network Model for Robot Control
=======================================
A neural network that learns to map (state, target) -> wheel velocities
using imitation learning from expert demonstrations.
"""

import numpy as np
import torch
import torch.nn as nn


class NavigationNet(nn.Module):
    """
    Neural network for robot navigation control.

    Input: [x, y, theta, x_target, y_target] - 5 dimensional
    Output: [v_left, v_right] - 2 dimensional (wheel velocities)
    """

    def __init__(self, input_dim: int = 4, hidden_dims: list = [64, 64], output_dim: int = 2):
        """
        Initialize the navigation network.

        Args:
            input_dim: Dimension of input (default: 5)
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output (default: 2)
        """
        super().__init__()

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

        # Store config
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class LargerNavigationNet(nn.Module):
    """
    A larger network variant for comparison.

    This network is more powerful but may overfit on small datasets.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: list = [128, 128, 64],
        output_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def normalize_inputs(inputs: np.ndarray, stats: dict | None = None) -> tuple:
    """
    Normalize inputs to zero mean and unit variance.

    Args:
        inputs: Input array of shape (N, 5) [x, y, theta, x_target, y_target]
        stats: Optional pre-computed statistics

    Returns:
        normalized_inputs, stats_dict
    """
    if stats is None:
        # Compute statistics
        mean = inputs.mean(axis=0)
        std = inputs.std(axis=0)

        # Avoid division by zero
        std[std < 1e-6] = 1.0

        stats = {"mean": mean, "std": std}

    normalized = (inputs - stats["mean"]) / stats["std"]

    return normalized, stats


class NavigationNetSimple(nn.Module):
    """
    Minimal network for simple navigation tasks.

    Sometimes simpler is better! This 2-layer network might be
    all you need for point-to-point navigation.
    """

    def __init__(self, max_velocity: float, input_dim: int = 4, output_dim: int = 2, hidden_dim: int = 32):
        super().__init__()
        self.max_velocity = max_velocity
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.net(compute_relative_features(x))
        return torch.tanh(out) * self.max_velocity


def compute_relative_features(inputs: torch.Tensor) -> torch.Tensor:
    """
    Convert raw state + target to relative input format.

    Args:
        x: Robot state [batch, 3] -> [x, y, theta]
        target: Target position [batch, 2] -> [x_target, y_target]

    Returns:
        Input tensor [batch, 4] -> [dx, dy, sin(theta), cos(theta)]
    """
    x = inputs[:, 0:1].clone()
    y = inputs[:, 1:2].clone()
    theta = inputs[:, 2:3].clone()
    x_targets = inputs[:, 3:4].clone()
    y_targets = inputs[:, 4:5].clone()

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    return torch.cat([x - x_targets, y - y_targets, sin_theta, cos_theta], dim=1)


def create_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: "default", "large", or "relative"
        **kwargs: Additional arguments passed to model constructor
    """
    if model_type == "default":
        return NavigationNet(**kwargs)
    elif model_type == "large":
        return LargerNavigationNet(**kwargs)
    elif model_type == "relative":
        assert "max_velocity" in kwargs, "max_velocity is required for relative model"  
        max_velocity = kwargs.pop("max_velocity")
        assert isinstance(max_velocity, float), "max_velocity must be a float"
        assert max_velocity > 0, "max_velocity must be positive"
        return NavigationNetSimple(max_velocity=max_velocity, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("Testing model architectures...\n")

    # Default model
    model = NavigationNet()
    print("NavigationNet:")
    print(f"  Input dim: {model.input_dim}")
    print(f"  Output dim: {model.output_dim}")
    print(f"  Parameters: {count_parameters(model)}")

    print("\n" + "=" * 50 + "\n")

    # Larger model
    model_large = LargerNavigationNet()
    print("LargerNavigationNet:")
    print(f"  Parameters: {count_parameters(model_large)}")

    print("\n" + "=" * 50 + "\n")

    # Angle-aware model
    model_angle = NavigationNetSimple(max_velocity=0.1)
    print("NavigationNetWithAngle:")
    print(f"  Input dim: {model_angle.input_dim}")
    print(f"  Parameters: {count_parameters(model_angle)}")