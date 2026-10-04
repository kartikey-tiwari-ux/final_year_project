"""Project-wide reproducibility helper. Call set_seed() at the start of any
training/evaluation script before any data splitting, model init, or sampling."""
import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
