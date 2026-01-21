import random
from pathlib import Path

import numpy as np
import torch

DATA_PATH = Path("data")
CHECKPOINTS_PATH = Path("checkpoints")
FIGURES_PATH = Path("figures")
TRAIN_FIGURES_PATH = Path("figures/train")
EVAL_FIGURES_PATH = Path("figures/eval")
DATASET_FIGURES_PATH = Path("figures/dataset")
ROBOT_FIGURES_PATH = Path("figures/robot")

DATA_PATH.mkdir(exist_ok=True)
CHECKPOINTS_PATH.mkdir(exist_ok=True)
FIGURES_PATH.mkdir(exist_ok=True)
TRAIN_FIGURES_PATH.mkdir(exist_ok=True)
EVAL_FIGURES_PATH.mkdir(exist_ok=True)
DATASET_FIGURES_PATH.mkdir(exist_ok=True)
ROBOT_FIGURES_PATH.mkdir(exist_ok=True)
THRESHOLD = 0.01


def seed_everything(seed: int):
    """Set the random seed for all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
