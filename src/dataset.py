"""
Dataset generation for robot navigation imitation learning.

Generates (state, target) -> control pairs from expert trajectories.
"""

import pickle
import random
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tomli

from model import compute_relative_features
from robot_simulator import (
    DifferentialDriveRobot,
    ExpertController,
    compute_distance_to_target,
    parse_config_robot,
    visualize_trajectory,
)
from utils import DATA_PATH, DATASET_FIGURES_PATH, seed_everything


def generate_random_scenario(difficulty: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random start and target positions for given difficulty.

    Args:
        difficulty: "easy", "medium", or "hard"

    Returns:
        (start, target): start [x, y, theta], target [x_target, y_target]
    """
    if difficulty == "easy":
        # Start near origin, target nearby
        target = np.array([np.random.uniform(0.5, 1.5), np.random.uniform(-0.5, 0.5)])
        start = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0])
        start[2] = np.arctan2(target[1] - start[1], target[0] - start[0]) + np.random.uniform(
            -np.pi / 4, np.pi / 4
        )
    elif difficulty == "medium":
        # Moderate distances and angles
        target = np.array([np.random.uniform(1, 2), np.random.uniform(-1, 1)])
        start = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0])
        start[2] = np.arctan2(target[1] - start[1], target[0] - start[0]) + np.random.uniform(
            -np.pi / 2, np.pi / 2
        )
    elif difficulty == "hard":
        # Long distances, any angle, targets can be anywhere
        target = np.array(
            [
                np.random.uniform(-3, 3),
                np.random.uniform(-3, 3),
            ]
        )
        start = np.array([np.random.uniform(-2, 2), np.random.uniform(-2, 2), 0])
        start[2] = np.arctan2(target[1] - start[1], target[0] - start[0]) + np.random.uniform(
            -np.pi, np.pi
        )
    else:
        raise ValueError(f"Invalid difficulty: {difficulty}")
    return start, target


def generate_dataset(expert: ExpertController, n_trajectories: int, include_hard: bool, max_steps: int, dt: float) -> Dict:
    """
    Generate dataset from expert trajectories.

    Args:
        expert: Expert controller for generating trajectories
        n_trajectories: Number of trajectories to generate
        include_hard: If True, include hard difficulty scenarios
        max_steps: Maximum steps per trajectory
        dt: Time step

    Returns:
        Dictionary with keys: 'inputs' (N, 5), 'outputs' (N, 2),
        'trajectory_ids' (N,), 'metadata' (list)
    """

    all_inputs = []
    all_outputs = []
    trajectory_ids = []
    metadata = []

    difficulties = ["easy", "medium"]
    if include_hard:
        difficulties.append("hard")

    for traj_id in range(n_trajectories):
        # Select difficulty
        difficulty = difficulties[traj_id % len(difficulties)]

        # Generate scenario
        start, target = generate_random_scenario(difficulty)

        # Generate expert trajectory
        states, controls = expert.generate_trajectory(start, target, max_steps, dt)

        # Create input-output pairs for each timestep
        for i, (state, control) in enumerate(zip(states[:-1], controls)):
            # Input: [x, y, theta, x_target, y_target]
            inp = np.concatenate([state, target])
            all_inputs.append(inp)
            all_outputs.append(control)
            trajectory_ids.append(traj_id)

        metadata.append(
            {
                "trajectory_id": traj_id,
                "start": start,
                "target": target,
                "difficulty": difficulty,
                "n_steps": len(controls),
                "final_distance": compute_distance_to_target(states[-1], target),
            }
        )

        if (traj_id + 1) % 100 == 0:
            print(f"Generated {traj_id + 1}/{n_trajectories} trajectories")

    return {
        "inputs": np.array(all_inputs),
        "outputs": np.array(all_outputs),
        "trajectory_ids": np.array(trajectory_ids),
        "metadata": metadata,
    }


def split_dataset(dataset: Dict, val_fraction: float) -> Tuple[Dict, Dict]:
    """
    Split dataset into training and validation sets by random sampling.

    Args:
        dataset: Full dataset dictionary
        val_fraction: Fraction of samples for validation (0-1)

    Returns:
        (train_dataset, val_dataset)
    """

    n_samples = len(dataset["inputs"])
    indices = np.arange(n_samples)

    np.random.shuffle(indices)

    split_idx = int(n_samples * (1 - val_fraction))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = {
        "inputs": dataset["inputs"][train_indices],
        "outputs": dataset["outputs"][train_indices],
        "trajectory_ids": dataset["trajectory_ids"][train_indices],
        "metadata": [
            dataset["metadata"][i] for i in np.unique(dataset["trajectory_ids"][train_indices])
        ],
    }

    val_dataset = {
        "inputs": dataset["inputs"][val_indices],
        "outputs": dataset["outputs"][val_indices],
        "trajectory_ids": dataset["trajectory_ids"][val_indices],
        "metadata": [
            dataset["metadata"][i] for i in np.unique(dataset["trajectory_ids"][val_indices])
        ],
    }

    return train_dataset, val_dataset


def split_dataset_by_trajectory(
    dataset: Dict, val_fraction: float
) -> Tuple[Dict, Dict]:
    """
    Split dataset by trajectories to avoid data leakage.

    Args:
        dataset: Full dataset dictionary
        val_fraction: Fraction of trajectories for validation (0-1)

    Returns:
        (train_dataset, val_dataset)
    """

    trajectory_ids = dataset["trajectory_ids"]
    unique_trajectories = np.unique(trajectory_ids)

    np.random.shuffle(unique_trajectories)

    split_idx = int(len(unique_trajectories) * (1 - val_fraction))
    train_trajectories = set(unique_trajectories[:split_idx])
    val_trajectories = set(unique_trajectories[split_idx:])

    train_mask = np.array([tid in train_trajectories for tid in trajectory_ids])
    val_mask = np.array([tid in val_trajectories for tid in trajectory_ids])

    train_dataset = {
        "inputs": dataset["inputs"][train_mask],
        "outputs": dataset["outputs"][train_mask],
        "trajectory_ids": dataset["trajectory_ids"][train_mask],
        "metadata": [
            dataset["metadata"][i] for i in np.unique(dataset["trajectory_ids"][train_mask])
        ],
    }

    val_dataset = {
        "inputs": dataset["inputs"][val_mask],
        "outputs": dataset["outputs"][val_mask],
        "trajectory_ids": dataset["trajectory_ids"][val_mask],
        "metadata": [
            dataset["metadata"][i] for i in np.unique(dataset["trajectory_ids"][val_mask])
        ],
    }

    return train_dataset, val_dataset


def create_dataloaders(train_dataset: Dict, val_dataset: Dict, batch_size: int):
    """
    Create PyTorch DataLoaders from dataset dictionaries.

    Args:
        train_dataset: Training dataset dictionary
        val_dataset: Validation dataset dictionary
        batch_size: Batch size for DataLoaders

    Returns:
        (train_loader, val_loader)
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Convert to tensors
    train_inputs = torch.FloatTensor(train_dataset["inputs"])
    train_outputs = torch.FloatTensor(train_dataset["outputs"])

    val_inputs = torch.FloatTensor(val_dataset["inputs"])
    val_outputs = torch.FloatTensor(val_dataset["outputs"])

    # Create datasets
    train_ds = TensorDataset(train_inputs, train_outputs)
    val_ds = TensorDataset(val_inputs, val_outputs)

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def save_dataset(dataset: Dict, filepath: str):
    """Save dataset dictionary to pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {filepath}")


def load_dataset(filepath: str) -> Dict:
    """Load dataset dictionary from pickle file."""
    with open(filepath, "rb") as f:
        dataset = pickle.load(f)
    return dataset


def print_dataset_stats(dataset: Dict, name: str):
    """Print dataset statistics: sample count, shapes, ranges, mean distance."""
    inputs = dataset["inputs"]
    outputs = dataset["outputs"]
    mean_distance = np.mean([metadata["final_distance"] for metadata in dataset["metadata"]])

    print(f"\n{name} Statistics:")
    print(f"  Number of samples: {len(inputs)}")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Output shape: {outputs.shape}")

    print("\n  Input ranges:")
    labels = ["x", "y", "theta", "x_target", "y_target"]
    for i, label in enumerate(labels):
        print(f"    {label}: [{inputs[:, i].min():.3f}, {inputs[:, i].max():.3f}]")

    print("\n  Output ranges:")
    labels = ["v_left", "v_right"]
    for i, label in enumerate(labels):
        print(f"    {label}: [{outputs[:, i].min():.3f}, {outputs[:, i].max():.3f}]")

    print(f"\n  Mean final distance: {mean_distance:.3f}")


def parse_config_dataset(config_dict: dict) -> dict:
    """Parse dataset configuration from TOML config dictionary."""
    seed = config_dict["dataset"]["seed"]
    n_trajectories = config_dict["dataset"]["n_trajectories"]
    include_hard = config_dict["dataset"]["include_hard"]
    val_fraction = config_dict["dataset"]["val_fraction"]
    realistic_robot = config_dict["dataset"]["realistic_robot"]
    return n_trajectories, include_hard, seed, val_fraction, realistic_robot

def main():
    import argparse

    # parse args from toml config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_robot", type=str, default="configs/robo_simulator_default.toml")
    parser.add_argument("--config_dataset", type=str, default="configs/dataset_default.toml")
    args = parser.parse_args()
    args = vars(args)

    # robot file
    with open(args["config_robot"], "rb") as f:
        config_robot = tomli.load(f)
    params, dt, max_steps = parse_config_robot(config_robot)

    # dataset file
    with open(args["config_dataset"], "rb") as f:
        config_dataset = tomli.load(f)
    n_trajectories, include_hard, seed, val_fraction, realistic_robot = parse_config_dataset(config_dataset)


    seed_everything(seed)
    robot = DifferentialDriveRobot(params=params, realistic=realistic_robot)
    expert = ExpertController(params=params, robot=robot)


    # Generate full dataset
    print("Generating robot navigation dataset...")
    dataset = generate_dataset(expert=expert, n_trajectories=n_trajectories, include_hard=include_hard, max_steps=max_steps, dt=dt)
    print_dataset_stats(dataset, "Full Dataset")

    # verify dataset graphs on robot_simulator.py using visualize_trajectory function
    fig, axes = plt.subplots(6, 3, figsize=(15, 30))
    axes = axes.flatten()
    idxs = random.sample(range(len(dataset["metadata"])), 18)
    for idx, ax in enumerate(axes):
        i = idxs[idx]
        target = dataset["metadata"][i]["target"]
        states = dataset["inputs"][dataset["trajectory_ids"] == i][:, :3].reshape(-1, 3)
        visualize_trajectory(
            states,
            target,
            title=f"Trajectory {i}, distance {dataset['metadata'][i]['final_distance']:.2f}",
            ax=ax,
        )
    plt.tight_layout()
    plt.savefig(str(DATASET_FIGURES_PATH / "dataset_trajectories.png"))
    print(f"Dataset trajectories saved to {DATASET_FIGURES_PATH / 'dataset_trajectories.png'}")

    # Split into train/val
    train_dataset, val_dataset = split_dataset(dataset, val_fraction=val_fraction)

    print_dataset_stats(train_dataset, "Training Set")
    print_dataset_stats(val_dataset, "Validation Set")

    # Save datasets
    save_dataset(dataset, str(DATA_PATH / "full_dataset.pkl"))
    save_dataset(train_dataset, str(DATA_PATH / "train_dataset.pkl"))
    save_dataset(val_dataset, str(DATA_PATH / "val_dataset.pkl"))

    print("\nDataset generation complete!")

if __name__ == "__main__":
    main()
