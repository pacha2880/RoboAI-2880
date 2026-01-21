# 🤖 Robot Navigation Challenge

A technical assessment for AI + Robotics engineering positions. Train a neural network to control a simulated differential-drive robot using imitation learning.

## 📋 Overview

In this challenge, you'll work with a codebase that trains a neural network to navigate a two-wheeled robot to target positions. The code has issues that cause poor real-world performance — your task is to investigate, diagnose, and fix them.

**Time Estimate:** 3.5 - 4.5 hours

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/gabo-di/RoboAI.git
cd RoboAI

# Create and activate virtual environment
conda create --name roboai python=3.11
conda activate roboai

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Step 1: Train the model
python src/train.py

# Step 2: Evaluate performance
python src/evaluate.py
```

## 📁 Project Structure

```
robot-nav-challenge/
├── src/                             # Source code
│   ├── robot_simulator.py           # Robot physics simulation
│   ├── dataset.py                   # Data generation & splitting
│   ├── model.py                     # Neural network architectures
│   ├── utis.py                      # Some utils
│   ├── train.py                     # Training script
│   └── evaluate.py                  # Evaluation script
├── tests/                           # Unit tests
├── configs/                         # Configuration files
│   ├── dataset_default.toml         # Dataset default configs
│   ├── evaluate_default.toml        # Evaluation default configs
│   ├── robo_simulator_default.toml  # Robo simulator default configs
│   └── train_default.toml           # Train default configs
├── docs/                            # Documentation
│   ├── PROBLEM_STATEMENT.md
│   ├── SOLUTION_REPORT_TEMPLATE.md
│   └── SIDE_QUESTS.md
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
├── requirements.txt
└── README.md
```

## 📖 Challenge Instructions

**Read the full problem statement:** [docs/PROBLEM_STATEMENT.md](docs/PROBLEM_STATEMENT.md)

**Optional bonus challenges:** [docs/SIDE_QUESTS.md](docs/SIDE_QUESTS.md)

## 🔧 For Candidates

### Submission Guidelines

1. **Fork this repository** to your own GitHub account
2. **Create a feature branch** for your work:
   ```bash
   git checkout -b solution/firstname-lastname
   ```
3. **Make your changes** with clear, atomic commits
4. **Document your findings** in `docs/SOLUTION_REPORT.md`
5. **Push your branch** and create a Pull Request

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add sin/cos angle preprocessing
fix: correct train/val data leakage
docs: add solution report
test: add unit tests for robot simulator
refactor: extract motor calibration to config
```

### What We're Looking For

- Systematic debugging approach
- Clear understanding of ML and robotics concepts
- Quality of implemented solutions
- Code quality and documentation
- Bonus: Creative solutions that surprise us!

## 📊 Expected Output

After training, you should see:
- Training curves saved to `training_curves.png`
- Model checkpoint in `checkpoints/best_model.pt`
- Evaluation results comparing ideal vs realistic robot performance

## 🐳 Docker (Optional Bonus)

See [docs/SIDE_QUESTS.md](docs/SIDE_QUESTS.md) for bonus challenges including Docker containerization.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ❓ Questions?

If something is unclear about the challenge requirements, please open an issue with the `question` label.

---

**Good luck!** 🍀
