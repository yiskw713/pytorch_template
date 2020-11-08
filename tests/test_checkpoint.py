import copy
import os
from typing import Tuple

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from src.libs.checkpoint import resume, save_checkpoint


@pytest.fixture()
def model_optim() -> Tuple[nn.Module, optim.Optimizer]:
    model = models.resnet18()
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    return model, optimizer


def test_checkpoint(model_optim: Tuple[nn.Module, optim.Optimizer]) -> None:
    model = model_optim[0]
    optimizer = model_optim[1]
    epoch = 100
    best_loss = 0.1
    result_path = "./tests/tmp"

    save_checkpoint(result_path, epoch, model, optimizer, best_loss)
    checkpoint_path = os.path.join(result_path, "checkpoint.pth")

    assert os.path.exists(checkpoint_path)

    model2 = copy.deepcopy(model)
    optimizer2 = copy.deepcopy(optimizer)

    begin_epoch, model2, optimizer2, best_loss2 = resume(
        checkpoint_path, model2, optimizer2
    )

    assert epoch == begin_epoch

    for state, state2 in zip(optimizer.state_dict(), optimizer2.state_dict()):
        assert state == state2

    assert best_loss == best_loss2

    # check if models have the same weights
    # https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351
    for key_item1, key_item2 in zip(
        model.state_dict().items(), model2.state_dict().items()
    ):
        assert torch.equal(key_item1[1], key_item2[1])

    os.remove(checkpoint_path)

    assert not os.path.exists(checkpoint_path)
