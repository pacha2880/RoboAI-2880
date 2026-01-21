"""
Training script for robot navigation model.

Trains neural network to predict wheel velocities from state and target.
"""

import json
import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tomli
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import (
    create_dataloaders,
    generate_dataset,
    parse_config_dataset,
    print_dataset_stats,
    save_dataset,
    split_dataset,
)
from model import count_parameters, create_model, normalize_inputs
from robot_simulator import (
    DifferentialDriveRobot,
    ExpertController,
    compute_distance_to_target,
    parse_config_robot,
    visualize_trajectory,
)
from utils import CHECKPOINTS_PATH, DATA_PATH, THRESHOLD, TRAIN_FIGURES_PATH, seed_everything


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    model_type: str,
) -> float:
    """
    Train model for one epoch.

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for inputs, targets in train_loader:
        inputs = inputs.to("cpu")
        targets = targets.to("cpu")

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    model_type: str,
) -> float:
    """
    Evaluate model on validation set.

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to("cpu")
            targets = targets.to("cpu")

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def train_model(
    model: nn.Module,
    train_dataset: Dict,
    val_dataset: Dict,
    n_epochs: int,
    lr: float,
    batch_size: int,
    max_rollouts: int,
    max_steps: int,
    dt: float,
    model_type: str,
    robot: DifferentialDriveRobot,
) -> Dict:
    """
    Full training loop with validation and rollouts.

    Args:
        model: Neural network model
        train_dataset: Training dataset dictionary
        val_dataset: Validation dataset dictionary
        n_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        max_rollouts: Maximum rollouts for visualization
        max_steps: Maximum simulation steps
        dt: Time step
        model_type: Model type string
        robot: Robot simulator instance

    Returns:
        Training history dictionary
    """
    # Normalize inputs
    train_inputs_norm, stats = normalize_inputs(train_dataset["inputs"])
    val_inputs_norm, _ = normalize_inputs(val_dataset["inputs"], stats)

    # Save normalization stats
    np.savez(DATA_PATH / "normalization_stats.npz", mean=stats["mean"], std=stats["std"])

    # Update datasets with normalized inputs
    train_dataset_norm = {
        "inputs": train_inputs_norm,
        "outputs": train_dataset["outputs"],
        "trajectory_ids": train_dataset["trajectory_ids"],
    }
    val_dataset_norm = {
        "inputs": val_inputs_norm,
        "outputs": val_dataset["outputs"],
        "trajectory_ids": val_dataset["trajectory_ids"],
    }

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_dataset_norm, val_dataset_norm, batch_size=batch_size
    )

    model = model.to("cpu")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    save_dir = CHECKPOINTS_PATH

    history = {"train_loss": [], "val_loss": [], "best_val_loss": float("inf"), "best_epoch": 0}

    print(f"\nTraining for {n_epochs} epochs...")
    print(f"Model parameters: {count_parameters(model)}")
    print(f"Model type: {model_type}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print()

    epoch_start_time = time.perf_counter()
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, model_type=model_type)
        val_loss = evaluate(model, val_loader, criterion, model_type=model_type)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        # Save best model
        if val_loss < history["best_val_loss"]:
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch
            torch.save(model.state_dict(), f"{save_dir}/best_model.pt")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    epoch_time = time.perf_counter() - epoch_start_time
    print(f"\nEpoch loop completed in {epoch_time:.2f} seconds")

    print("\nRolling out model on train dataset...")
    rollout_start_time = time.perf_counter()
    rollout_model_on_dataset(
        model,
        train_dataset,
        mode_type="train",
        max_rollouts=max_rollouts,
        model_type=model_type,
        robot=robot,
        max_steps=max_steps,
        dt=dt,
    )
    train_rollout_time = time.perf_counter() - rollout_start_time
    print(f"Train rollout completed in {train_rollout_time:.2f} seconds")

    print("\nRolling out model on validation dataset...")
    rollout_start_time = time.perf_counter()
    rollout_model_on_dataset(
        model,
        val_dataset,
        mode_type="val",
        max_rollouts=max_rollouts,
        model_type=model_type,
        robot=robot,
        max_steps=max_steps,
        dt=dt,
    )
    val_rollout_time = time.perf_counter() - rollout_start_time
    print(f"Validation rollout completed in {val_rollout_time:.2f} seconds")

    print(
        f"\nBest validation loss: {history['best_val_loss']:.6f} at epoch {history['best_epoch'] + 1}"
    )

    # Save final model and history
    torch.save(model.state_dict(), f"{save_dir}/final_model.pt")
    with open(f"{save_dir}/history.json", "w") as f:
        json.dump(history, f)

    return history


def rollout_model_on_dataset(
    model: nn.Module,
    dataset: Dict,
    mode_type: str,
    max_rollouts: int,
    model_type: str,
    robot: DifferentialDriveRobot,
    max_steps: int,
    dt: float,
) -> None:
    """
    Rollout trained model on dataset trajectories and visualize results.

    Args:
        model: Trained model
        dataset: Dataset dictionary with metadata
        mode_type: "train" or "val" for filename
        max_rollouts: Maximum number of trajectories to rollout
        model_type: Model type string
        robot: Robot simulator instance
        max_steps: Maximum simulation steps
        dt: Time step
    """
    # Load the normalization stats used during training
    # This is critical - the model expects normalized inputs!
    stats_path = DATA_PATH / "normalization_stats.npz"
    if stats_path.exists():
        stats_data = np.load(stats_path)
        stats = {"mean": stats_data["mean"], "std": stats_data["std"]}
    else:
        # Fallback to identity if stats don't exist (shouldn't happen)
        print("Warning: normalization_stats.npz not found, using identity normalization")
        stats = {
            "mean": np.zeros(5),  # x, y, theta, x_target, y_target
            "std": np.ones(5),
        }

    outputs = []
    trajectory_ids = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, metadata in enumerate(dataset["metadata"]):
        trajectory_id = metadata["trajectory_id"]
        if trajectory_id not in trajectory_ids:
            targets = metadata["target"]
            start = metadata["start"]
            robot.reset_state(*start)
            states, controls, reached = run_model_on_robot(
                model, robot, targets, stats, max_steps=max_steps, model_type=model_type, dt=dt
            )
            outputs.append(reached)
            trajectory_ids.append(metadata["trajectory_id"])
            if idx < len(axes):
                visualize_trajectory(
                    states,
                    targets,
                    title=f"Trajectory {trajectory_id} success: {reached}",
                    ax=axes[idx],
                )
            if len(trajectory_ids) >= max_rollouts:
                break

    plt.tight_layout()
    plt.savefig(str(TRAIN_FIGURES_PATH / f"rollout_model_on_dataset_{mode_type}.png"))
    print(f"Rollout model on dataset trajectory ids: {trajectory_ids}")
    print(f"Rollout model on dataset len success: {len(outputs)}")
    print(f"Rollout model on dataset mean success: {np.mean(outputs)}")
    print(f"Rollout model on dataset std success: {np.std(outputs)}")


def run_model_on_robot(
    model: nn.Module,
    robot: DifferentialDriveRobot,
    target: np.ndarray,
    stats: dict,
    max_steps: int,
    dt: float,
    model_type: str,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Run trained model to control robot from current state to target.

    Args:
        model: Trained navigation model
        robot: Robot simulator instance
        target: Target position [x, y]
        stats: Normalization statistics
        max_steps: Maximum simulation steps
        dt: Time step
        model_type: Model type string

    Returns:
        (states, controls, reached): trajectory, controls, success flag
    """
    states = [robot.get_state()]
    controls = []

    model.eval()
    with torch.no_grad():
        for step in range(max_steps):
            state = robot.get_state()

            # Check if reached target
            dist = compute_distance_to_target(state, target)
            if dist < THRESHOLD:
                return np.array(states), np.array(controls), True

            # Prepare input: [x, y, theta, x_target, y_target]
            inp = np.concatenate([state, target])

            # Normalize
            inp_norm = (inp - stats["mean"]) / stats["std"]

            # Forward pass
            inp_tensor = torch.FloatTensor(inp_norm).unsqueeze(0).to("cpu")
            output = model(inp_tensor)

            v_left, v_right = output[0].cpu().numpy()

            # Apply control to robot
            controls.append([v_left, v_right])
            robot.step(v_left, v_right, dt=dt)
            states.append(robot.get_state())

    return np.array(states), np.array(controls), False


def plot_training_history(history: Dict, save_path: str):
    """Plot training and validation loss curves with best epoch marker."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "b-", label="Training Loss")
    ax.plot(epochs, history["val_loss"], "r-", label="Validation Loss")

    ax.axvline(
        history["best_epoch"] + 1,
        color="g",
        linestyle="--",
        label=f"Best Epoch ({history['best_epoch'] + 1})",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training Progress")
    ax.legend()
    ax.grid(True)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(str(save_path))
    print(f"Training curves saved to {save_path}")


def parse_config_train(config_train: dict) -> dict:
    """Parse training configuration from TOML config dictionary."""
    batch_size = config_train["train"]["batch_size"]
    n_epochs = config_train["train"]["n_epochs"]
    lr = config_train["train"]["lr"]
    seed = config_train["train"]["seed"]
    max_rollouts = config_train["train"]["max_rollouts"]
    model_type = config_train["train"]["model_type"]
    return batch_size, n_epochs, lr, seed, max_rollouts, model_type


def main():
    """Main training function with dataset generation, training, and evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Train robot navigation model")

    parser.add_argument("--config_train", type=str, default="configs/train_default.toml")
    parser.add_argument("--config_robot", type=str, default="configs/robo_simulator_default.toml")
    parser.add_argument("--config_dataset", type=str, default="configs/dataset_default.toml")
    args = parser.parse_args()
    args = vars(args)

    # train file
    with open(args["config_train"], "rb") as f:
        config_train = tomli.load(f)
    batch_size, n_epochs, lr, seed, max_rollouts, model_type = parse_config_train(config_train)

    # robot file
    with open(args["config_robot"], "rb") as f:
        config_robot = tomli.load(f)
    params, dt, max_steps = parse_config_robot(config_robot)

    # dataset file
    with open(args["config_dataset"], "rb") as f:
        config_dataset = tomli.load(f)
    n_trajectories, include_hard, seed, val_fraction, realistic_robot = parse_config_dataset(
        config_dataset
    )

    seed_everything(seed)
    robot = DifferentialDriveRobot(params=params, realistic=realistic_robot)
    expert = ExpertController(params=params, robot=robot)

    # Generate full dataset
    print("Generating robot navigation dataset...")
    dataset_start_time = time.perf_counter()
    dataset = generate_dataset(
        expert=expert,
        n_trajectories=n_trajectories,
        include_hard=include_hard,
        max_steps=max_steps,
        dt=dt,
    )
    dataset_time = time.perf_counter() - dataset_start_time
    print(f"Dataset generation completed in {dataset_time:.2f} seconds")
    print_dataset_stats(dataset, "Full Dataset")
    train_dataset, val_dataset = split_dataset(dataset, val_fraction=val_fraction)
    save_dataset(train_dataset, str(DATA_PATH / "train_dataset.pkl"))
    save_dataset(val_dataset, str(DATA_PATH / "val_dataset.pkl"))

    print(f"Training samples: {len(train_dataset['inputs'])}")
    print(f"Validation samples: {len(val_dataset['inputs'])}")
    print_dataset_stats(train_dataset, "Train Dataset")
    print_dataset_stats(val_dataset, "Val Dataset")

    # Create model
    model = create_model(model_type=model_type)

    # Train
    history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        max_rollouts=max_rollouts,
        model_type=model_type,
        robot=robot,
        max_steps=max_steps,
        dt=dt,
    )

    # Plot results
    plot_training_history(history, save_path=str(TRAIN_FIGURES_PATH / "training_curves.png"))

    print("\nTraining complete!")
    print(f"Model saved to {CHECKPOINTS_PATH}/best_model.pt")


if __name__ == "__main__":
    main()
