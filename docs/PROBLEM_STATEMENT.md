# 🤖 Robot Navigation Challenge

## Problem Description

Welcome to the technical assessment for the AI Robotics Engineer position!

You are given a codebase that trains a neural network to control a **differential-drive robot** for navigation tasks. The robot has two wheels and must learn to navigate from any starting position to a target location.

### The System

**Robot:** A two-wheeled differential-drive robot (think Roomba)
- State: `[x, y, θ]` — position (meters) and heading (radians)
- Control: `[v_left, v_right]` — left and right wheel velocities (m/s)

**Learning Approach:** Imitation Learning (Behavior Cloning)
- An expert controller generates optimal trajectories
- A neural network learns to mimic the expert
- Input: `[x, y, θ, x_target, y_target]`
- Output: `[v_left, v_right]`

### The Problem

When we train this model, it shows good training metrics, but **performs poorly during actual robot evaluation**. Your task is to investigate why the trained model fails and propose/implement fixes.

---

## 🎯 Your Mission

### Part 1: Investigation (45-60 min recommended)

Run the provided code and investigate the problem.

#### Step 1: Setup
```bash
# Create and activate virtual environment
conda create --name roboai python=3.11
conda activate roboai

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Robot movement simulation
```bash
python src/robot_simulator.py
```

You should see:
- The logs on terminal
- Training curves in `figures/robot`


#### Step 3: Dataset generation
```bash
python src/dataset.py
```

You should see:
- The logs on terminal
- Training curves in `figures/dataset`


#### Step 4: Train
```bash
python src/train.py
```

You should see:
- The logs on terminal
- Training curves in `figures/train`

#### Step 5: Evaluate
```bash
python src/evaluate.py
```

You should see:
- The logs on terminal
- Training curves in `figures/eval`


#### Questions to Explore
- What could cause this the bad results on logs and figures?
- Are there issues with the data? The model? The physics?
- How would you systematically diagnose the problem?

### Part 2: Fix and Improve (2-3 hrs recommended)

Propose and implement fixes for the issues you discovered.

**You may modify:**
- Dataset generation or splitting
- Model architecture
- Training procedure
- Physics simulation
- Preprocessing/postprocessing
- Anything else you think helps!

**Document your approach:**
- What hypotheses did you form?
- What experiments did you run?
- What did you learn?

---

## 📊 Evaluation Criteria

| Criterion | Weight | What We're Looking For |
|-----------|--------|------------------------|
| **Systematic Debugging** | 30% | Methodical investigation, clear hypotheses |
| **Technical Understanding** | 25% | Grasp of ML concepts, robotics, physics |
| **Solution Quality** | 25% | Effective fixes, clean implementation |
| **Communication** | 20% | Clear explanation of findings and approach |

**Bonus:** Solutions or insights that make us say "I wouldn't have thought of that!"

---

## 📁 File Overview

| File | Purpose |
|------|---------|
| `src/robot_simulator.py` | Differential-drive robot physics |
| `src/dataset.py` | Expert trajectories, train/val splitting |
| `src/model.py` | Neural network architecture |
| `src/train.py` | Training loop |
| `src/evaluate.py` | Evaluation and visualization |

---

## 📝 Deliverables

When you're done, prepare:

### 1. Solution Report
Create `docs/SOLUTION_REPORT.md` (template provided) describing:
- Issues you discovered
- Your debugging process
- Solutions you implemented
- Results after fixes
- Remaining issues or future improvements

### 2. Modified Code
- Commit your changes with clear messages
- Follow the branching guidelines in `CONTRIBUTING.md`

### 3. Be Prepared to Discuss
- Your debugging thought process
- Tradeoffs in your solutions
- What you'd do differently with more time

---

## ⏱️ Time Management

| Task | Suggested Time |
|------|----------------|
| Setup & initial run | 10-15 min |
| Investigation | 30-45 min |
| Implementation | 30-45 min |
| Documentation | 15-20 min |
| **Total** | **1.5-2 hours** |

**Note:** You're not expected to find and fix everything. Quality of investigation matters more than quantity of fixes.

---

## 🎮 Bonus: Side Quests

See [SIDE_QUESTS.md](SIDE_QUESTS.md) for optional bonus challenges:
- Docker containerization
- Numerical precision improvements
- Video generation
- Unit testing
- And more!

---

## ❓ Questions?

If something is unclear, open a GitHub issue with the `question` label.

---

**Good luck!** 🍀🤖
