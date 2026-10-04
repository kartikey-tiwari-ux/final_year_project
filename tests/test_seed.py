import random

import numpy as np

from src.utils.seed import set_seed


def test_set_seed_reproducible_python_random():
    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_reproducible_numpy():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_set_seed_different_seeds_differ():
    set_seed(1)
    a = [random.random() for _ in range(5)]
    set_seed(2)
    b = [random.random() for _ in range(5)]
    assert a != b
