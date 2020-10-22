import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    result_path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    best_loss: float,
) -> None:

    save_states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
    }

    torch.save(save_states, os.path.join(result_path, "checkpoint.pth"))


def resume(
    resume_path: str, model: nn.Module, optimizer: optim.Optimizer
) -> Tuple[int, nn.Module, optim.Optimizer, nn.Module, float]:

    assert os.path.exists(resume_path), "there is no checkpoint at the result folder"

    print("loading checkpoint {}".format(resume_path))
    checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)

    begin_epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    model.load_state_dict(checkpoint["state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    print("training will start from {} epoch".format(begin_epoch))

    return begin_epoch, model, optimizer, best_loss
