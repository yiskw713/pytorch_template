import random

import torch

from src.libs.seed import set_seed


def test_get_model() -> None:
    set_seed(seed=0)
    value1 = random.random()
    tensor1 = torch.randn(1, 2, 3)

    set_seed(seed=0)
    value2 = random.random()
    tensor2 = torch.randn(1, 2, 3)

    assert value1 == value2
    assert torch.all(tensor1 == tensor2)
