"""
Evaluation script for robot navigation model.

Evaluates trained model by running it on simulated robot scenarios.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tomli
import torch
import torch.nn as nn

from dataset import generate_random_scenario
from model import create_model
from robot_simulator import (
    DifferentialDriveRobot,
    RobotParams,
    compute_distance_to_target,
    parse_config_robot,
    visualize_trajectory,
)
from train import run_model_on_robot
from utils import CHECKPOINTS_PATH, DATA_PATH, EVAL_FIGURES_PATH, seed_everything


def load_model(checkpoint_path: str, model_type: str, max_velocity) -> nn.Module:
    """Load trained model from checkpoint file."""
    model = create_model(model_type=model_type, max_velocity=max_velocity)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def load_normalization_stats(stats_path: str) -> dict:
    """Load normalization statistics from npz file."""
    data = np.load(stats_path)
    return {"mean": data["mean"], "std": data["std"]}


def evaluate_on_scenarios(
    model: nn.Module,
    d_max: float,
    n_scenarios: int,
    use_realistic_robot: bool,
    scenario_difficulty: str,
    seed: int,
    model_type: str,
    params: RobotParams,
    dt: float,
    max_steps: int,
) -> dict:

    """
    Evaluate model on multiple random scenarios.

    Args:
        model: Trained model
        stats: Normalization statistics
        n_scenarios: Number of test scenarios
        use_realistic_robot: If True, use robot with motor imperfections
        scenario_difficulty: "easy", "medium", or "hard"
        seed: Random seed
        model_type: Model type string
        params: Robot physical parameters
        dt: Time step
        max_steps: Maximum simulation steps

    Returns:
        Dictionary with keys: successes, failures, final_distances, scenarios
    """
    np.random.seed(seed)

    results = {"successes": 0, "failures": 0, "final_distances": [], "scenarios": []}
    robot = DifferentialDriveRobot(params=params, realistic=use_realistic_robot)

    for i in range(n_scenarios):
        # Generate random scenario
        start, target = generate_random_scenario(scenario_difficulty)

        # Create robot
        robot.reset_state(*start)

        # Run model
        states, controls, reached = run_model_on_robot(
            model,
            robot,
            target,
            d_max=d_max,
            model_type=model_type,
            dt=dt,
            max_steps=max_steps,
        )

        final_dist = compute_distance_to_target(states[-1], target)

        results["final_distances"].append(final_dist)
        if reached:
            results["successes"] += 1
        else:
            results["failures"] += 1

        results["scenarios"].append(
            {
                "start": start.tolist(),
                "target": target.tolist(),
                "states": states,
                "controls": controls,
                "reached": reached,
                "final_distance": final_dist,
            }
        )

    return results


def visualize_evaluation_results(results: dict, save_path: str):
    """Plot and save visualization of evaluation trajectory results."""
    n_scenarios = len(results["scenarios"])

    # Select a few scenarios to visualize
    n_show = min(6, n_scenarios)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < n_show:
            scenario = results["scenarios"][idx]
            states = np.array(scenario["states"])
            target = np.array(scenario["target"])

            status = (
                "SUCCESS"
                if scenario["reached"]
                else f"FAILED (dist={scenario['final_distance']:.2f})"
            )
            visualize_trajectory(states, target, title=f"Scenario {idx + 1}: {status}", ax=ax)
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(EVAL_FIGURES_PATH / save_path))
    print(f"Evaluation visualization saved to {EVAL_FIGURES_PATH / save_path}")


def print_evaluation_summary(results: dict, robot_type: str):
    """Print summary of evaluation results."""
    print(f"\n{'=' * 50}")
    print(f"Evaluation Results ({robot_type})")
    print(f"{'=' * 50}")
    print(f"Total scenarios: {results['successes'] + results['failures']}")
    print(f"Successes: {results['successes']}")
    print(f"Failures: {results['failures']}")
    print(
        f"Success rate: {results['successes'] / (results['successes'] + results['failures']):.2f}"
    )
    print("\nFinal distance statistics:")
    print(f"  Mean: {np.mean(results['final_distances']):.3f}")
    print(f"  Std:  {np.std(results['final_distances']):.3f}")
    print(f"  Min:  {np.min(results['final_distances']):.3f}")
    print(f"  Max:  {np.max(results['final_distances']):.3f}")

def parse_config_evaluate(config_evaluate: dict) -> tuple:
    """Parse evaluation configuration from TOML config dictionary."""
    n_scenarios = config_evaluate["eval"]["n_scenarios"]
    scenario_difficulty = config_evaluate["eval"]["scenario_difficulty"]
    seed = config_evaluate["eval"]["seed"]
    model_type = config_evaluate["eval"]["model_type"]
    return n_scenarios, scenario_difficulty, seed, model_type

def main():
    """Main evaluation function with ideal and realistic robot testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate robot navigation model")

    parser.add_argument("--config_robot", type=str, default="configs/robo_simulator_default.toml")
    parser.add_argument("--config_evaluate", type=str, default="configs/evaluate_default.toml")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINTS_PATH / "best_model.pt"))
    args = parser.parse_args()

    # Check if model exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Model checkpoint not found at {args.checkpoint}")
        print("Please train a model first using: python train.py")
        return

    # parse args from toml config file
    with open(args.config_robot, "rb") as f:
        config_robot = tomli.load(f)
    params, dt, max_steps = parse_config_robot(config_robot)
    with open(args.config_evaluate, "rb") as f:
        config_evaluate = tomli.load(f)
    n_scenarios, scenario_difficulty, seed, model_type = parse_config_evaluate(config_evaluate)
    seed_everything(seed)

    # Load model and stats
    print("Loading model...")
    model = load_model(args.checkpoint, model_type=model_type, max_velocity=params.max_v)
    stats_path = DATA_PATH / "relative_feature_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError("relative_feature_stats.npz not found. Train the model first.")

    d_max = float(np.load(stats_path)["d_max"])
    print(f"[eval] Loaded d_max = {d_max:.4f}")


    time_start = time.perf_counter()
    print("\nEvaluating on idealized robot")
    results_ideal = evaluate_on_scenarios(
        model,
        d_max=d_max,
        n_scenarios=n_scenarios,
        use_realistic_robot=False,
        scenario_difficulty=scenario_difficulty,
        seed=seed,
        model_type=model_type,
        params=params,
        dt=dt,
        max_steps=max_steps,
    )
    time_ideal = time.perf_counter() - time_start
    print_evaluation_summary(results_ideal, "Idealized Robot")
    print(f"Idealized robot evaluation completed in {time_ideal:.2f} seconds")


    time_start = time.perf_counter()
    print("\nEvaluating on realistic robot")
    results_real = evaluate_on_scenarios(
        model,
        d_max=d_max,
        n_scenarios=n_scenarios,
        use_realistic_robot=False,
        scenario_difficulty=scenario_difficulty,
        seed=seed,
        model_type=model_type,
        params=params,
        dt=dt,
        max_steps=max_steps,
    )
    time_real = time.perf_counter() - time_start
    print_evaluation_summary(results_real, "Realistic Robot")
    print(f"Realistic robot evaluation completed in {time_real:.2f} seconds")

    # Visualize results
    visualize_evaluation_results(results_real, "evaluation_realistic.png")
    visualize_evaluation_results(results_ideal, "evaluation_ideal.png")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    ideal_rate = results_ideal["successes"] / (
        results_ideal["successes"] + results_ideal["failures"]
    )
    real_rate = results_real["successes"] / (results_real["successes"] + results_real["failures"])

    print(f"Idealized robot success rate: {ideal_rate:.2f}")
    print(f"Realistic robot success rate: {real_rate:.2f}")

    if real_rate < ideal_rate - 0.1:
        print("\n⚠️  WARNING: Large performance gap between ideal and realistic robots!")
        print("   There might be a sim-to-real gap issue to investigate.")


if __name__ == "__main__":
    main()
