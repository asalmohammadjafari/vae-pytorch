from pathlib import Path
import random
from typing import Union

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Union[str, Path]):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(state: dict, path: Union[str, Path]):
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(path: Union[str, Path], device: torch.device):
    return torch.load(path, map_location=device)
